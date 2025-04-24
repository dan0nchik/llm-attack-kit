from tqdm import tqdm
import textgrad as tg
import numpy as np
import config
import pandas as pd
from textgrad.autograd.string_based_ops import StringBasedFunction
from textgrad.engine_experimental.litellm import LiteLLMEngine
import wandb
from helper_functions import *


wandb.login()
set_seed(12)

os.environ["OLLAMA_BASE_URL"] = config.OLLAMA_BASE_URL
llm_api_eval = tg.get_engine(config.BACKWARD_MODEL)
if "ollama" in config.BASE_MODEL:
    llm_api_test = tg.get_engine(config.BASE_MODEL)
else:
    llm_api_test = LiteLLMEngine(config.BASE_MODEL)
tg.set_backward_engine(llm_api_eval, override=True)

# Load the data and the evaluation function
train_set = JBB("train")
val_set = JBB("val")
test_set = JBB("test")
eval_fn = StringBasedFunction(string_based_equality_fn, function_purpose=fn_purpose)
# train_set, val_set, test_set, eval_fn = load_task("BBH_object_counting", evaluation_api=llm_api_eval)
print("Train/Val/Test Set Lengths: ", len(train_set), len(val_set), len(test_set))
STARTING_SYSTEM_PROMPT = train_set.get_task_description()


logs_table = wandb.Table(
    columns=[
        "epoch",
        "step",
        "test_acc",
        "validation_acc",
        "prompt",
        "goal",
        "target",
        "answer",
    ]
)

run = wandb.init(
    # Set the wandb project where this run will be logged.
    project="textgrad-optimize",
    # Track hyperparameters and run metadata.
    config={
        "optimizer_prompt": OPTIMIZER_SYSTEM_PROMPT,
        "starting_prompt": STARTING_SYSTEM_PROMPT,
        "attacker_model": BACKWARD_MODEL,
        "base_model": BASE_MODEL,
        "epochs": NUM_EPOCHS,
        "steps": NUM_STEPS_PER_EPOCH,
    },
)

train_loader = tg.tasks.DataLoader(train_set, batch_size=3, shuffle=True)


# Testing the 0-shot performance of the evaluation engine
system_prompt = tg.Variable(
    STARTING_SYSTEM_PROMPT,
    requires_grad=True,
    role_description="system prompt to the language model",
)
model_evaluation = tg.BlackboxLLM(llm_api_eval, system_prompt)

system_prompt = tg.Variable(
    STARTING_SYSTEM_PROMPT,
    requires_grad=True,
    role_description="system prompt to the language model",
)
model = tg.BlackboxLLM(llm_api_test, system_prompt)

optimizer = tg.TextualGradientDescent(
    engine=llm_api_eval,
    parameters=[system_prompt],
    optimizer_system_prompt=OPTIMIZER_SYSTEM_PROMPT,
)

test_acc = eval_dataset(test_set, eval_fn, model)
validation_acc = eval_dataset(val_set, eval_fn, model)
prompt = system_prompt.get_value()

# log locally
results = {"answer": [], "test_acc": [], "prompt": [], "validation_acc": []}
results["test_acc"].append(test_acc)
results["validation_acc"].append(validation_acc)
results["prompt"].append(prompt)

# log wandb
logs_table.add_data(
    -1,
    -1,
    np.mean(test_acc),
    np.mean(validation_acc),
    prompt,
    "zero-shot",
    "zero-shot",
    "zero-shot",
)

for epoch in range(config.NUM_EPOCHS):
    for steps, (batch_x, batch_y) in enumerate(
        (pbar := tqdm(train_loader, position=0))
    ):
        pbar.set_description(f"Training step {steps}. Epoch {epoch}")
        optimizer.zero_grad()
        losses = []
        for x, y in zip(batch_x, batch_y):
            x = tg.Variable(
                x, requires_grad=False, role_description="query to the language model"
            )
            y = tg.Variable(
                y, requires_grad=False, role_description="correct answer for the query"
            )
            response = model(x)
            logs_table.add_data(
                steps,
                epoch,
                np.mean(test_acc),
                np.mean(validation_acc),
                system_prompt.get_value(),
                x.get_value(),
                y.get_value(),
                response.get_value(),
            )
            try:
                eval_output_variable = eval_fn(
                    inputs=dict(prediction=response, ground_truth_answer=y)
                )
            except:
                eval_output_variable = eval_fn([x, y, response])
            losses.append(eval_output_variable)
        total_loss = tg.sum(losses)
        total_loss.backward()

        try:
            optimizer.step()
        except Exception as e:
            print("Error in optimizer step: ", e)
            logs_table.add_data(
                steps,
                epoch,
                np.mean(test_acc),
                np.mean(validation_acc),
                e,
                "ERROR",
                "ERROR",
                "ERROR",
            )
            if steps == NUM_STEPS_PER_EPOCH:
                break
            continue
        run_validation_revert(system_prompt, results, model, eval_fn, val_set)
        print("=" * 50)
        print("sys prompt: ", system_prompt)
        test_acc = eval_dataset(test_set, eval_fn, model)
        validation_acc = results["validation_acc"][-1]
        results["test_acc"].append(test_acc)
        results["prompt"].append(system_prompt.get_value())

        logs_table.add_data(
            steps,
            epoch,
            np.mean(test_acc),
            np.mean(validation_acc),
            system_prompt.get_value(),
            "EVALUATION",
            "EVALUATION",
            "EVALUATION",
        )
        if steps == NUM_STEPS_PER_EPOCH:
            break

wandb.log({"logs_table": logs_table})
run.finish()

# df = pd.DataFrame(logs_table.data, columns=logs_table.columns)

# df["validation_acc"].plot()

# running the best system prompt on the whole dataset

splits = {
    "harmful": "data/harmful-behaviors.csv",
    "benign": "data/benign-behaviors.csv",
}
df = pd.read_csv("hf://datasets/JailbreakBench/JBB-Behaviors/" + splits["harmful"])

best_system_prompt = results["prompt"][-1]

answers = []

model = tg.BlackboxLLM(llm_api_test, best_system_prompt)

for goal in df["Goal"]:
    answers.append(
        model(
            tg.Variable(
                goal,
                requires_grad=False,
                role_description="query to the language model",
            )
        )
    )

print("ASR:", len(list(filter(lambda x: "Step" in x.value, answers))))
