import os
from typing import Union
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import StructuredTool
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_core.callbacks import BaseCallbackHandler
from pydantic import BaseModel, Field

# Mock Fraud Database
FRAUD_DATABASE = {
    "U001": {"flagged": True,  "reason": "Multiple identity fraud reports in 2023"},
    "U002": {"flagged": False, "reason": "Clean record"},
    "U123": {"flagged": False, "reason": "Clean record"},
    "U456": {"flagged": True,  "reason": "Suspected synthetic identity — mismatched SSN patterns"},
    "U789": {"flagged": False, "reason": "Clean record"},
    "U999": {"flagged": True,  "reason": "Previous loan default with fraudulent documentation"},
}

# Pydantic schemas — accept str/int/float from LLM
class CreditScoreInput(BaseModel):
    score: Union[str, int, float] = Field(..., description="Credit score integer between 300 and 850")

class IncomeValidatorInput(BaseModel):
    input_str: str = Field(..., description="Monthly income and loan amount as 'monthly_income, loan_amount' e.g. '45000, 200000'")

class FraudCheckerInput(BaseModel):
    user_id: Union[str, int] = Field(..., description="User ID string e.g. U123")

class LoanDecisionInput(BaseModel):
    input_str: str = Field(..., description="All three results concatenated: credit check output, income validator output, and fraud check output as one string")

# Tool implementations
def _credit_score_checker(score: Union[str, int, float]) -> str:
    score = int(float(str(score).replace(",", "")))
    if score < 580:
        category, note = "Poor", "High default risk. Lender likely to decline."
    elif score < 670:
        category, note = "Fair", "Moderate risk. Approval possible with conditions."
    elif score < 740:
        category, note = "Good", "Low risk. Generally qualifies for standard rates."
    else:
        category, note = "Excellent", "Very low risk. Qualifies for best rates."
    return f"Credit Risk Category: {category} (Score: {score}). {note}"


def _income_validator(input_str: str) -> str:
    import re
    nums = re.findall(r"[\d]+(?:[.,]\d+)*", input_str.replace("$", ""))
    nums = [float(n.replace(",", "")) for n in nums]
    if len(nums) < 2:
        return "Error: could not parse monthly_income and loan_amount from input."
    monthly_income, loan_amount = nums[0], nums[1]
    monthly_rate = 0.05 / 12
    n_payments = 360
    monthly_payment = loan_amount * (monthly_rate * (1 + monthly_rate) ** n_payments) /                       ((1 + monthly_rate) ** n_payments - 1)
    dti = (monthly_payment / monthly_income) * 100
    if dti < 28:
        verdict = "ACCEPTABLE — DTI well within safe lending limits."
    elif dti < 36:
        verdict = "BORDERLINE — DTI is manageable but on the higher side."
    elif dti < 43:
        verdict = "HIGH — DTI exceeds recommended threshold; elevated risk."
    else:
        verdict = "UNACCEPTABLE — DTI too high; standard loan approval unlikely."
    return (
        f"Monthly Income: ${monthly_income:,.2f} | "
        f"Estimated Monthly Payment: ${monthly_payment:,.2f} | "
        f"DTI Ratio: {dti:.1f}% | Verdict: {verdict}"
    )


def _fraud_flag_checker(user_id: Union[str, int]) -> str:
    user_id = str(user_id).strip().upper()
    record = FRAUD_DATABASE.get(user_id)
    if record is None:
        return f"User ID '{user_id}' not found in fraud database. Treating as CLEAN — no prior fraud history on record."
    if record["flagged"]:
        return f"FRAUD FLAG DETECTED for {user_id}: {record['reason']}. Immediate escalation recommended."
    return f"No fraud flags for {user_id}. {record['reason']}."


def _loan_decision_maker(input_str: str) -> str:
    blob = input_str.lower()
    reject_reasons, approve_notes = [], []
    if "fraud flag detected" in blob:
        reject_reasons.append("Fraud flag on applicant's record.")
    if "poor" in blob:
        reject_reasons.append("Credit score falls in 'Poor' category (below 580).")
    elif "fair" in blob:
        reject_reasons.append("Credit score is 'Fair' — elevated risk; conditional approval only.")
    elif "good" in blob:
        approve_notes.append("Credit score is 'Good' — acceptable risk.")
    elif "excellent" in blob:
        approve_notes.append("Credit score is 'Excellent' — premium applicant.")
    if "unacceptable" in blob:
        reject_reasons.append("Debt-to-income ratio is unacceptable (>43%).")
    elif "dti ratio is high" in blob or "high — dti exceeds" in blob:
        reject_reasons.append("DTI ratio is high (36–43%), exceeding recommended limits.")
    elif "borderline" in blob:
        approve_notes.append("DTI is borderline but within manageable range.")
    elif "acceptable" in blob:
        approve_notes.append("DTI ratio is well within safe lending limits.")
    if reject_reasons:
        return (
            f"DECISION: REJECT\n"
            f"Reasons: {' | '.join(reject_reasons)}\n"
            f"The application does not meet minimum lending criteria."
        )
    return (
        f"DECISION: APPROVE\n"
        f"Basis: {(' | '.join(approve_notes)) or 'All criteria met.'}\n"
        f"The applicant satisfies all core lending requirements."
    )

# Wire up StructuredTools with explicit schemas
credit_score_checker = StructuredTool.from_function(
    func=_credit_score_checker,
    name="credit_score_checker",
    description="Evaluates a credit score (300-850) and returns Poor/Fair/Good/Excellent risk category.",
    args_schema=CreditScoreInput,
)

income_validator = StructuredTool.from_function(
    func=_income_validator,
    name="income_validator",
    description="Calculates debt-to-income ratio. Pass input_str as 'monthly_income loan_amount' e.g. '45000 200000'.",
    args_schema=IncomeValidatorInput,
)

fraud_flag_checker = StructuredTool.from_function(
    func=_fraud_flag_checker,
    name="fraud_flag_checker",
    description="Checks a user_id string against the fraud database. Returns fraud status.",
    args_schema=FraudCheckerInput,
)

loan_decision_maker = StructuredTool.from_function(
    func=_loan_decision_maker,
    name="loan_decision_maker",
    description="Makes final APPROVE/REJECT decision. Pass input_str as one combined string containing all three previous tool outputs: credit check result, income validator result, and fraud check result.",
    args_schema=LoanDecisionInput,
)

TOOLS = [credit_score_checker, income_validator, fraud_flag_checker, loan_decision_maker]

REACT_PROMPT = PromptTemplate.from_template("""
You are an expert loan underwriting AI agent. Your job is to evaluate loan applications by using available tools in a logical sequence.

Always follow this order:
1. Check the credit score first.
2. Validate income and DTI ratio.
3. Check for fraud flags.
4. Make the final loan decision using all three results.

Think step-by-step. After gathering all information, call loan_decision_maker with the results.

You have access to the following tools:
{tools}

Use the following format:
Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought: {agent_scratchpad}
""")


def build_agent(groq_api_key: str, callbacks=None):
    """Builds and returns the LangChain ReAct AgentExecutor."""
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        groq_api_key=groq_api_key,
        callbacks=callbacks or [],
    )

    agent = create_react_agent(llm=llm, tools=TOOLS, prompt=REACT_PROMPT)
    executor = AgentExecutor(
        agent=agent,
        tools=TOOLS,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=10,
        callbacks=callbacks or [],
    )
    return executor


def run_loan_agent(user_id: str, credit_score: int, monthly_income: float,
                   loan_amount: float, groq_api_key: str, callbacks=None):
    """
    Entry point: runs the ReAct agent for a loan application.
    Returns the agent's output dict.
    """
    query = (
        f"Evaluate this loan application — "
        f"User ID: {user_id}, "
        f"Credit Score: {credit_score}, "
        f"Monthly Income: {monthly_income}, "
        f"Loan Amount: {loan_amount}"
    )
    executor = build_agent(groq_api_key, callbacks=callbacks)
    result = executor.invoke({"input": query})
    return result