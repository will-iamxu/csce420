# analysis.py
# -----------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


######################
# ANALYSIS QUESTIONS #
######################

# Set the given parameters to obtain the specified policies through
# value iteration.

def question2():
    answerDiscount = 0.9
    answerNoise = 0.0
    return answerDiscount, answerNoise

def question3a():
    # Prefer close exit (+1), risking cliff (-10)
    # Low discount => prefer nearby rewards; zero noise => no risk of falling off cliff
    answerDiscount = 0.1
    answerNoise = 0.0
    answerLivingReward = 0.0
    return answerDiscount, answerNoise, answerLivingReward

def question3b():
    # Prefer close exit (+1), avoiding cliff (-10)
    # Low discount => prefer nearby rewards; some noise => cliff is genuinely risky
    answerDiscount = 0.1
    answerNoise = 0.1
    answerLivingReward = 0.0
    return answerDiscount, answerNoise, answerLivingReward

def question3c():
    # Prefer distant exit (+10), risking cliff (-10)
    # High discount => value distant reward; zero noise => safe to take short cliff-adjacent path
    answerDiscount = 0.9
    answerNoise = 0.0
    answerLivingReward = 0.0
    return answerDiscount, answerNoise, answerLivingReward

def question3d():
    # Prefer distant exit (+10), avoiding cliff (-10)
    # High discount => value distant reward; some noise => cliff is risky so take safe upper path
    answerDiscount = 0.9
    answerNoise = 0.2
    answerLivingReward = 0.0
    return answerDiscount, answerNoise, answerLivingReward

def question3e():
    # Avoid both exits and cliff (episode never terminates)
    # Large positive living reward => agent prefers to keep living over any exit
    answerDiscount = 0.9
    answerNoise = 0.0
    answerLivingReward = 100.0
    return answerDiscount, answerNoise, answerLivingReward

def question8():
    # No epsilon/lr combination guarantees >99% success crossing narrow bridge in 50 episodes
    return 'NOT POSSIBLE'

if __name__ == '__main__':
    print('Answers to analysis questions:')
    import analysis
    for q in [q for q in dir(analysis) if q.startswith('question')]:
        response = getattr(analysis, q)()
        print('  Question %s:\t%s' % (q, str(response)))
