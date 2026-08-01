#include "detection/ai_aimbot/ai_aimbot_module.h"

#include "detection/detection_system.h"
#include "movement_analysis/player_context.h"
#include "movement/movement.h"
#include "sdk/usercmd.h"

#include <algorithm>
#include <cmath>
#include <limits>

CConVar<bool> cs2ac_ai_aimbot_debug("cs2ac_ai_aimbot_debug", FCVAR_NONE, "Show why AI Aimbot accepts or rejects each damaging shot", false);

#define AI_AIMBOT_DEBUG(...) \
	do \
	{ \
		if (cs2ac_ai_aimbot_debug.GetBool()) \
			Msg("[CS2AC AI Aimbot] " __VA_ARGS__); \
	} while (0)

namespace
{
	constexpr size_t sampleHistorySize = 32;
	constexpr size_t signatureHistorySize = 8;
	constexpr int detectionScore = 12;
	constexpr auto evidenceWindow = std::chrono::minutes(10);
	constexpr int minTrajectoryDeltas = 6;

	// Metric 1: smoothness — coefficient of variation of per-tick angular delta magnitudes.
	// Human aim shows natural micro-jitter; a Bezier or logistic-eased curve yields nearly uniform magnitudes.
	constexpr float smoothnessMinMean = 1.0f;
	constexpr float smoothnessMaxCV = 0.15f;
	constexpr int smoothnessPoints = 3;

	// Metric 2: constant angular velocity — longest run of consecutive near-equal magnitudes.
	// Linear interpolation on the aim curve produces long runs of identical per-tick speed; muscle motion does not.
	constexpr int constantVelocityMinRun = 4;
	constexpr float constantVelocityMaxDeviation = 0.15f;
	constexpr float constantVelocityMinSpeed = 5.0f;
	constexpr int constantVelocityPoints = 2;

	// Metric 3: repeated curve signature — cosine similarity of the pre-shot delta vector against prior shots.
	// The same easing function fired twice produces near-identical shapes; a human never repeats a curve exactly.
	constexpr float signatureMinMagnitude = 8.0f;
	constexpr float signatureMinCosine = 0.97f;
	constexpr int signaturePoints = 5;

	// Metric 4: subtick singleton — a shot command whose aim delta is concentrated in the firing subtick.
	// AI aimbots that write only the "final" angle before pulling the trigger leave every other subtick empty.
	constexpr float subtickMinTotal = 5.0f;
	constexpr float subtickSingleRatio = 0.90f;
	constexpr int subtickPoints = 4;

	float YawDelta(float from, float to)
	{
		return std::remainder(to - from, 360.0f);
	}
} // namespace

namespace detection
{
	void AiAimbotModule::Load(Announce announceCallback, ShotCorrelator *shotCorrelator)
	{
		announce = announceCallback;
		shots = shotCorrelator;
	}

	void AiAimbotModule::Unload()
	{
		Reset();
		shots = nullptr;
		announce = nullptr;
	}

	void AiAimbotModule::Reset()
	{
		playerData = {};
	}

	void AiAimbotModule::OnProcessUsercmds(MovementPlayer *player, PlayerCommand *commands, int numCommands)
	{
		if (!IsEligibleHuman(player) || !commands || numCommands <= 0)
		{
			return;
		}

		auto &data = playerData[player->index];
		for (int i = 0; i < numCommands; ++i)
		{
			PlayerCommand &command = commands[i];
			if (!command.has_base() || !command.base().has_viewangles())
			{
				continue;
			}
			if (std::any_of(data.samples.rbegin(), data.samples.rend(),
							[&](const AiAimAngleSample &stored) { return stored.commandNumber == command.cmdNum; }))
			{
				continue;
			}

			const auto &base = command.base();
			AiAimAngleSample sample;
			sample.commandNumber = command.cmdNum;
			sample.clientTick = base.client_tick();
			sample.angles = QAngle(base.viewangles().x(), base.viewangles().y(), base.viewangles().z());
			if (!IsFinite(sample.angles))
			{
				continue;
			}

			sample.hasAttack = command.attack1_start_history_index() >= 0;

			for (int moveIndex = 0; moveIndex < base.subtick_moves_size(); ++moveIndex)
			{
				const auto &move = base.subtick_moves(moveIndex);
				float pitchDelta = 0.0f;
				float yawDelta = 0.0f;
				if (move.has_pitch_delta() && std::isfinite(move.pitch_delta()))
				{
					pitchDelta = move.pitch_delta();
				}
				if (move.has_yaw_delta() && std::isfinite(move.yaw_delta()))
				{
					yawDelta = move.yaw_delta();
				}
				const float combinedAbs = std::abs(pitchDelta) + std::abs(yawDelta);
				sample.subtickTotalAbs += combinedAbs;
				sample.subtickDominantAbs = (std::max)(sample.subtickDominantAbs, combinedAbs);
				++sample.subtickMoveCount;
			}

			data.samples.push_back(sample);
			while (data.samples.size() > sampleHistorySize)
			{
				data.samples.pop_front();
			}
		}
	}

	void AiAimbotModule::OnSetupMove(MovementPlayer *player, PlayerCommand *command, int currentTick)
	{
		if (!IsEligibleHuman(player) || !command)
		{
			return;
		}

		auto &data = playerData[player->index];
		auto found = std::find_if(data.samples.rbegin(), data.samples.rend(),
								  [&](AiAimAngleSample &stored) { return stored.commandNumber == command->cmdNum; });
		if (found == data.samples.rend())
		{
			return;
		}
		found->simulated = true;
		found->serverTick = currentTick;
		if (data.pending)
		{
			Evaluate(player, data, currentTick);
		}
	}

	void AiAimbotModule::OnPlayerHurt(MovementPlayer *attacker, MovementPlayer *victim, ShotRecord &shot)
	{
		if (!IsEligibleHuman(attacker) || !victim || attacker == victim || shot.playerIndex != attacker->index)
		{
			return;
		}
		auto &data = playerData[attacker->index];
		if (data.pending)
		{
			Evaluate(attacker, data, shot.fireTick);
			data.pending = false;
		}
		data.pendingShot = shot.commandNumber;
		data.pendingShotTick = shot.fireTick;
		data.victimIndex = victim->index;
		data.pending = true;
		Evaluate(attacker, data, shot.fireTick);
	}

	void AiAimbotModule::OnGameFrame(int currentTick)
	{
		for (int index = 1; index <= MAXPLAYERS; ++index)
		{
			auto &data = playerData[index];
			if (!data.pending)
			{
				continue;
			}
			auto *player = g_pCS2ACPlayerManager ? g_pCS2ACPlayerManager->ToPlayer(static_cast<u32>(index)) : nullptr;
			if (!IsEligibleHuman(player))
			{
				data.pending = false;
				data.pendingShot = -1;
				data.pendingShotTick = -1;
				continue;
			}
			Evaluate(player, data, currentTick);
		}
	}

	void AiAimbotModule::Evaluate(MovementPlayer *player, AiAimPlayerData &data, int currentTick)
	{
		if (!data.pending)
		{
			return;
		}
		auto clearPending = [&]()
		{
			data.pending = false;
			data.pendingShot = -1;
			data.pendingShotTick = -1;
			data.victimIndex = -1;
		};

		auto shot = std::find_if(data.samples.begin(), data.samples.end(),
								 [&](const AiAimAngleSample &stored) { return stored.commandNumber == data.pendingShot && stored.simulated; });
		if (shot == data.samples.end())
		{
			if (static_cast<std::int64_t>(currentTick) - data.pendingShotTick <= 2)
			{
				return;
			}
			AI_AIMBOT_DEBUG("%s pending shot %d was never simulated within the grace window.\n", player->GetName(), data.pendingShot);
			clearPending();
			return;
		}
		if (data.lastEvaluatedCommand == shot->commandNumber)
		{
			clearPending();
			return;
		}
		data.lastEvaluatedCommand = shot->commandNumber;

		std::array<const AiAimAngleSample *, kAiAimbotSignatureLength + 1> trajectory {};
		int count = 0;
		int wantedTick = shot->serverTick;
		for (auto it = data.samples.rbegin(); it != data.samples.rend() && count < static_cast<int>(trajectory.size()); ++it)
		{
			if (!it->simulated || it->serverTick > wantedTick)
			{
				continue;
			}
			if (it->serverTick < wantedTick)
			{
				break;
			}
			trajectory[count++] = &*it;
			--wantedTick;
		}
		if (count < minTrajectoryDeltas + 1)
		{
			AI_AIMBOT_DEBUG("%s only %d consecutive-tick samples were available for shot %d.\n", player->GetName(), count, shot->commandNumber);
			clearPending();
			return;
		}

		const int deltaCount = count - 1;
		std::array<float, kAiAimbotSignatureLength> yawDeltas {};
		std::array<float, kAiAimbotSignatureLength> pitchDeltas {};
		std::array<float, kAiAimbotSignatureLength> magnitudes {};
		for (int i = 0; i < deltaCount; ++i)
		{
			const auto *newer = trajectory[i];
			const auto *older = trajectory[i + 1];
			yawDeltas[i] = YawDelta(older->angles.y, newer->angles.y);
			pitchDeltas[i] = newer->angles.x - older->angles.x;
			magnitudes[i] = std::sqrt(yawDeltas[i] * yawDeltas[i] + pitchDeltas[i] * pitchDeltas[i]);
		}

		if (data.lastSmoothnessCommand != shot->commandNumber)
		{
			float sum = 0.0f;
			for (int i = 0; i < deltaCount; ++i)
			{
				sum += magnitudes[i];
			}
			const float mean = sum / deltaCount;
			if (mean > smoothnessMinMean)
			{
				float variance = 0.0f;
				for (int i = 0; i < deltaCount; ++i)
				{
					variance += (magnitudes[i] - mean) * (magnitudes[i] - mean);
				}
				const float stdDev = std::sqrt(variance / deltaCount);
				const float cv = stdDev / mean;
				if (std::isfinite(cv) && cv < smoothnessMaxCV)
				{
					data.lastSmoothnessCommand = shot->commandNumber;
					AI_AIMBOT_DEBUG("%s smoothness: CV %.3f, mean %.2f deg/tick over %d ticks.\n", player->GetName(), cv, mean, deltaCount);
					const int total = RecordEvidence(data, smoothnessPoints);
					if (total >= detectionScore && announce)
					{
						announce("AI AIMBOT", player,
								 localization::Format("evidence.ai_aimbot.smoothness",
													  "Unnaturally smooth aim trajectory (coefficient of variation {cv} over {samples} ticks at {mean} "
													  "deg/tick) reached the AI-aimbot threshold {score}/{threshold}.",
													  {{"cv", tfm::format("%.3f", cv)},
													   {"samples", tfm::format("%d", deltaCount)},
													   {"mean", tfm::format("%.2f", mean)},
													   {"score", tfm::format("%d", total)},
													   {"threshold", tfm::format("%d", detectionScore)}}));
						data.evidence.clear();
						clearPending();
						return;
					}
				}
			}
		}

		if (data.lastConstantVelocityCommand != shot->commandNumber && deltaCount >= constantVelocityMinRun)
		{
			int longestRun = 1;
			int currentRun = 1;
			float bestRunSum = 0.0f;
			int bestRunLength = 0;
			float currentRunSum = magnitudes[0];
			for (int i = 1; i < deltaCount; ++i)
			{
				if (std::abs(magnitudes[i] - magnitudes[i - 1]) < constantVelocityMaxDeviation)
				{
					++currentRun;
					currentRunSum += magnitudes[i];
					if (currentRun > longestRun)
					{
						longestRun = currentRun;
						bestRunSum = currentRunSum;
						bestRunLength = currentRun;
					}
				}
				else
				{
					currentRun = 1;
					currentRunSum = magnitudes[i];
				}
			}
			if (longestRun >= constantVelocityMinRun && bestRunLength > 0)
			{
				const float avgSpeed = bestRunSum / bestRunLength;
				if (avgSpeed >= constantVelocityMinSpeed)
				{
					data.lastConstantVelocityCommand = shot->commandNumber;
					AI_AIMBOT_DEBUG("%s constant velocity: %d ticks at %.2f deg/tick.\n", player->GetName(), longestRun, avgSpeed);
					const int total = RecordEvidence(data, constantVelocityPoints);
					if (total >= detectionScore && announce)
					{
						announce("AI AIMBOT", player,
								 localization::Format("evidence.ai_aimbot.constant_velocity",
													  "Sustained constant angular velocity ({ticks} ticks at {speed} deg/tick) reached the AI-aimbot "
													  "threshold {score}/{threshold}.",
													  {{"ticks", tfm::format("%d", longestRun)},
													   {"speed", tfm::format("%.2f", avgSpeed)},
													   {"score", tfm::format("%d", total)},
													   {"threshold", tfm::format("%d", detectionScore)}}));
						data.evidence.clear();
						clearPending();
						return;
					}
				}
			}
		}

		AiAimShotSignature currentSig;
		currentSig.commandNumber = shot->commandNumber;
		currentSig.lastServerTick = shot->serverTick;
		const int sigDeltas = (std::min)(deltaCount, kAiAimbotSignatureLength);
		float magSquared = 0.0f;
		for (int i = 0; i < sigDeltas; ++i)
		{
			currentSig.yawDeltas[i] = yawDeltas[i];
			currentSig.pitchDeltas[i] = pitchDeltas[i];
			magSquared += yawDeltas[i] * yawDeltas[i] + pitchDeltas[i] * pitchDeltas[i];
		}
		currentSig.magnitude = std::sqrt(magSquared);
		currentSig.valid = std::isfinite(currentSig.magnitude) && currentSig.magnitude > 0.0f && sigDeltas == kAiAimbotSignatureLength;

		if (currentSig.valid && currentSig.magnitude >= signatureMinMagnitude && data.lastSignatureCommand != shot->commandNumber)
		{
			float bestCosine = 0.0f;
			int matchedCommand = -1;
			for (const auto &prior : data.shotSignatures)
			{
				if (!prior.valid || prior.magnitude < signatureMinMagnitude)
				{
					continue;
				}
				float dot = 0.0f;
				for (int i = 0; i < kAiAimbotSignatureLength; ++i)
				{
					dot += currentSig.yawDeltas[i] * prior.yawDeltas[i] + currentSig.pitchDeltas[i] * prior.pitchDeltas[i];
				}
				const float cosine = dot / (currentSig.magnitude * prior.magnitude);
				if (std::isfinite(cosine) && cosine > bestCosine)
				{
					bestCosine = cosine;
					matchedCommand = prior.commandNumber;
				}
			}
			if (bestCosine >= signatureMinCosine)
			{
				data.lastSignatureCommand = shot->commandNumber;
				AI_AIMBOT_DEBUG("%s repeated signature: cosine %.3f versus command %d.\n", player->GetName(), bestCosine, matchedCommand);
				const int total = RecordEvidence(data, signaturePoints);
				if (total >= detectionScore && announce)
				{
					announce("AI AIMBOT", player,
							 localization::Format("evidence.ai_aimbot.repeated_signature",
												  "Repeated aim-curve signature (cosine {cos} against a prior {length}-tick approach) reached the "
												  "AI-aimbot threshold {score}/{threshold}.",
												  {{"cos", tfm::format("%.3f", bestCosine)},
												   {"length", tfm::format("%d", kAiAimbotSignatureLength)},
												   {"score", tfm::format("%d", total)},
												   {"threshold", tfm::format("%d", detectionScore)}}));
					data.evidence.clear();
					if (currentSig.valid)
					{
						data.shotSignatures.push_back(currentSig);
						while (data.shotSignatures.size() > signatureHistorySize)
						{
							data.shotSignatures.pop_front();
						}
					}
					clearPending();
					return;
				}
			}
		}
		if (currentSig.valid)
		{
			data.shotSignatures.push_back(currentSig);
			while (data.shotSignatures.size() > signatureHistorySize)
			{
				data.shotSignatures.pop_front();
			}
		}

		if (data.lastSubtickCommand != shot->commandNumber && shot->hasAttack && shot->subtickMoveCount >= 2)
		{
			const float totalMotion = shot->subtickTotalAbs;
			const float dominantMotion = shot->subtickDominantAbs;
			if (totalMotion >= subtickMinTotal && dominantMotion > 0.0f && (dominantMotion / totalMotion) >= subtickSingleRatio)
			{
				data.lastSubtickCommand = shot->commandNumber;
				const float ratio = dominantMotion / totalMotion;
				AI_AIMBOT_DEBUG("%s subtick singleton: %.2f/%.2f (%.0f%%) written on a single subtick.\n", player->GetName(), dominantMotion, totalMotion,
								ratio * 100.0f);
				const int total = RecordEvidence(data, subtickPoints);
				if (total >= detectionScore && announce)
				{
					announce("AI AIMBOT", player,
							 localization::Format("evidence.ai_aimbot.subtick_singleton",
												  "Single-subtick aim input ({ratio}% of the shot command's aim delta was concentrated in one subtick) "
												  "reached the AI-aimbot threshold {score}/{threshold}.",
												  {{"ratio", tfm::format("%.0f", ratio * 100.0f)},
												   {"score", tfm::format("%d", total)},
												   {"threshold", tfm::format("%d", detectionScore)}}));
					data.evidence.clear();
				}
			}
		}

		clearPending();
	}

	int AiAimbotModule::RecordEvidence(AiAimPlayerData &data, int points)
	{
		const auto now = std::chrono::steady_clock::now();
		while (!data.evidence.empty() && now - data.evidence.front().time >= evidenceWindow)
		{
			data.evidence.pop_front();
		}
		data.evidence.push_back({now, points});
		int total = 0;
		for (const auto &incident : data.evidence)
		{
			total += incident.points;
		}
		return total;
	}

	void AiAimbotModule::OnClientDisconnect(MovementPlayer *player)
	{
		if (player && player->index >= 1 && player->index <= MAXPLAYERS)
		{
			playerData[player->index] = {};
		}
	}
} // namespace detection
