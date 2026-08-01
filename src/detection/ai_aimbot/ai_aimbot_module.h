#pragma once

#include "common.h"
#include "localization.h"
#include "sdk/datatypes.h"

#include <array>
#include <chrono>
#include <cstdint>
#include <deque>

class MovementPlayer;
class PlayerCommand;

namespace detection
{
	class ShotCorrelator;
	struct ShotRecord;

	inline constexpr int kAiAimbotSignatureLength = 8;

	struct AiAimAngleSample
	{
		int commandNumber {};
		int clientTick {};
		int serverTick {-1};
		QAngle angles;
		float subtickTotalAbs {};
		float subtickDominantAbs {};
		int subtickMoveCount {};
		bool hasAttack {};
		bool simulated {};
	};

	struct AiAimShotSignature
	{
		int commandNumber {};
		int lastServerTick {-1};
		std::array<float, kAiAimbotSignatureLength> yawDeltas {};
		std::array<float, kAiAimbotSignatureLength> pitchDeltas {};
		float magnitude {};
		bool valid {};
	};

	struct AiAimIncident
	{
		std::chrono::steady_clock::time_point time;
		int points {};
	};

	struct AiAimPlayerData
	{
		std::deque<AiAimAngleSample> samples;
		std::deque<AiAimShotSignature> shotSignatures;
		std::deque<AiAimIncident> evidence;
		int pendingShot {-1};
		int pendingShotTick {-1};
		int victimIndex {-1};
		int lastEvaluatedCommand {-1};
		int lastSmoothnessCommand {-1};
		int lastConstantVelocityCommand {-1};
		int lastSignatureCommand {-1};
		int lastSubtickCommand {-1};
		bool pending {};
	};

	// Detects AI-driven aim assist (humanized aimbots, ML-fit curves, easing-function aim) by analysing the
	// server-visible view-angle trajectory that leads into a damaging shot. Complements AimbotModule (which
	// catches snap-hits) and AimlockModule (which catches sustained tracking) with statistical signatures
	// only present in synthetic aim: unnaturally smooth trajectories, constant angular velocity runs,
	// repeated curve shapes across shots, and single-subtick "silent" writes.
	class AiAimbotModule
	{
	public:
		using Announce = void (*)(const char *detection, MovementPlayer *player, const localization::Text &evidence);

		void Load(Announce announceCallback, ShotCorrelator *shotCorrelator);
		void Unload();
		void Reset();
		void OnProcessUsercmds(MovementPlayer *player, PlayerCommand *commands, int numCommands);
		void OnSetupMove(MovementPlayer *player, PlayerCommand *command, int currentTick);
		void OnGameFrame(int currentTick);
		void OnPlayerHurt(MovementPlayer *attacker, MovementPlayer *victim, ShotRecord &shot);
		void OnClientDisconnect(MovementPlayer *player);

	private:
		void Evaluate(MovementPlayer *player, AiAimPlayerData &data, int currentTick);
		int RecordEvidence(AiAimPlayerData &data, int points);

		Announce announce {};
		ShotCorrelator *shots {};
		std::array<AiAimPlayerData, MAXPLAYERS + 1> playerData;
	};
} // namespace detection
