from evaluate import *

from game_config import *
from learning_config import *
import engine as eng
from drawable import *

from rlfb_qlearning import *
from nn import *
from models_evolutionary import *

print("Evaluating Evo1:", evaluate(to_eval_evo1))
print("Evaluating Evo2:", evaluate(to_eval_evo2))
print("Evaluating Evo3:", evaluate(to_eval_evo3))

model = Model_Q()
model.fit(20000)

print("Evaluating Q_model:", evaluate(model.action_evaluate))

model = Model_SARSA()
model.fit(20000)

print("Evaluating SARSA:", evaluate(model.action_evaluate))

model = Model_NN()
# model.fit(20000)
# model.save_model()
model.load_model()


print("Evaluating NN:", evaluate(model.action_evaluate))






