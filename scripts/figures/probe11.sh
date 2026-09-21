cd /data/doubleBee/doubleBee_terr_spawn
echo "########## eval_climb.py CLI options"
grep -n "add_argument" scripts/paper/eval_climb.py
echo
echo "########## how goals are sampled / any fixed-goal hook"
grep -rn -i "goal" scripts/paper/eval_climb.py | head -25
echo
echo "########## goal command/resample in the env"
grep -rn -i "goal_pos\|resample\|GOAL_FIXED\|DOUBLEBEE_GOAL" lab/doublebee/tasks/manager_based/locomotion/velocity/doublebee_env/mdp/commands*.py lab/doublebee/tasks/manager_based/locomotion/velocity/doublebee_env/mdp/*.py 2>/dev/null | grep -i "goal" | head -25
echo
echo "########## env vars that touch goals"
grep -rho "DOUBLEBEE_[A-Z0-9_]*GOAL[A-Z0-9_]*\|DOUBLEBEE_GOAL[A-Z0-9_]*\|DOUBLEBEE_TARGET[A-Z0-9_]*" lab scripts 2>/dev/null | sort -u
