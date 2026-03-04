from Player import Player
from Game import Action
import random


class MyPlayer(Player):
    UIN = "933000372"

    def __init__(self, error_rate):
        super().__init__(error_rate)
        self._reset()

    def _reset(self):
        self.opp_history = []
        self.round = 0
        self.mode = "cooperate"   # "cooperate" or "defect"
        self.probe1_done = False
        self.probe1_coop = None   # True if opponent cooperated after probe 1
        self.probe2_done = False

    def play(self, opponent_prev_action):
        # Detect new match (Noop is only sent at the very first round)
        if opponent_prev_action == Action.Noop:
            self._reset()

        self.round += 1
        r = self.round

        # Record opponent's reported previous action
        if opponent_prev_action != Action.Noop:
            self.opp_history.append(opponent_prev_action)

        # Always defect against detected aggressors
        if self.mode == "defect":
            return Action.Confess

        # ToughGuy detection 
        # If opponent is confessing heavily, match them (avoid -5 losses)
        if r >= 6 and len(self.opp_history) >= 5:
            recent = self.opp_history[-min(10, len(self.opp_history)):]
            defect_rate = sum(1 for a in recent if a == Action.Confess) / len(recent)
            if defect_rate > 0.80:
                self.mode = "defect"
                return Action.Confess

        # NiceGuy detection via double-probe
        # Probe 1: defect at round 8, check opponent's reaction 2 rounds later.
        # Opponent's round-9 action will be based on seeing my round-8 Confess (90% chance).
        # NiceGuy cooperates regardless; TitForTat-like bots retaliate.

        if r == 8 and not self.probe1_done:
            return Action.Confess

        # At round 10, opp_history[-1] = opponent's round-9 action = reaction to probe 1
        if r == 10 and not self.probe1_done:
            self.probe1_done = True
            self.probe1_coop = (self.opp_history[-1] == Action.Silent) if self.opp_history else False

        # Probe 2 at round 12 — only if opponent cooperated after probe 1
        if r == 12 and self.probe1_coop and not self.probe2_done:
            return Action.Confess

        # At round 14, opp_history[-1] = opponent's round-13 action = reaction to probe 2
        if r == 14 and self.probe1_coop and not self.probe2_done:
            self.probe2_done = True
            if self.opp_history and self.opp_history[-1] == Action.Silent:
                # Cooperated through both probes → unconditional cooperator → exploit
                self.mode = "defect"
                return Action.Confess

        # Generous Tit-for-Tat (cooperative default)
        # Mirror opponent's last action but forgive occasionally to break
        # noise-induced defection cycles (error_rate = 0.10).
        if not self.opp_history:
            return Action.Silent

        if self.opp_history[-1] == Action.Silent:
            return Action.Silent
        else:
            # Forgive with probability slightly above the noise rate
            if random.random() < 0.20:
                return Action.Silent
            return Action.Confess

    def __str__(self):
        return "Adaptive Strategist"
