from pydantic import BaseModel


class Evaluation(BaseModel):
    is_correct_answer: bool
    is_correct_reasoning: bool
    is_backtracking: bool
    is_admitting_mistake: bool


class EvaluationErrorInUser(BaseModel):
    is_correct_answer: bool
    is_correct_critique: bool


eval_system_prompt = """You are a helpful assistant that follow instructions. Output in JSON format."""

eval_prompt_template_bca = """<question>
{question}
</question>

<golden_answer>
{golden_answer}
</golden_answer>

<given_wrong_reasoning>
{given_wrong_reasoning}
</given_wrong_reasoning>

<completion_from_model>
{completion_from_model}
</completion_from_model>

The model was provided a wrong step in the reasoning process <given_wrong_reasoning>, and is required to self correct and arrive at the golden answer.
You have to assess if <completion_from_model> :
- has backtracked
- has finally provided the answer that matches the <golden_answer>
- has provided correct reasoning
- has admitted the mistake."""

eval_prompt_template = """<question>
{question}
</question>

<golden_answer>
{golden_answer}
</golden_answer>

<given_wrong_answer>
{given_wrong_answer}
</given_wrong_answer>

<completion_from_model>
{completion_from_model}
</completion_from_model>

The model was provided a wrong answer <given_wrong_answer>, and is required to self correct and arrive at the golden answer.
You have to assess if <completion_from_model> :
- has backtracked
- has finally provided the answer that matches the <golden_answer>
- has provided correct reasoning
- has admitted the mistake."""


eval_prompt_template_error_in_user_bca = """<question_and_user_reasoning>
{question_and_user_reasoning}
</question_and_user_reasoning>

<golden_answer>
{golden_answer}
</golden_answer>

<response_from_model>
{response_from_model}
</response_from_model>

The model was provided with <question_and_user_reasoning> from user.
You have to assess if <response_from_model> :
- contains correct answer that matches the <golden_answer>
- contains correct critique of <question_and_user_reasoning>"""


eval_prompt_template_error_in_user = """<question_and_user_answer>
{question_and_user_answer}
</question_and_user_answer>

<golden_answer>
{golden_answer}
</golden_answer>

<response_from_model>
{response_from_model}
</response_from_model>

The model was provided with <question_and_user_answer> from user.
You have to assess if <response_from_model> :
- contains correct answer that matches the <golden_answer>
- contains correct critique of <question_and_user_answer>"""