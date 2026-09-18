from langsmith import Client
from pydantic import BaseModel, Field
from typing import cast
from advanced_rag_agent.retrieval.hybrid_retriever import hybrid_search
from advanced_rag_agent.retrieval.reranker import rerank_documents
from advanced_rag_agent.retrieval.context_builder import build_context
from advanced_rag_agent.generation.answer_generator import generate_answer
from advanced_rag_agent.config.settings import TOP_K
from langchain_core.prompts import ChatPromptTemplate
from advanced_rag_agent.generation.llm import get_llm

DATASET_NAME = "agentic-rag-hr-evaluation"
EVAL_K = 3

client = Client()


def generation_target(inputs: dict) -> dict:
    question = inputs["question"]

    # Use the same retrieval pipeline as the production RAG
    retrieved_docs = hybrid_search(
        question,
        k=TOP_K,
    )

    reranked_docs = rerank_documents(
        question,
        retrieved_docs,
        top_k=EVAL_K,
    )

    # Generate the answer using the production RAG prompt
    answer = generate_answer(
        query=question,
        documents=reranked_docs,
    )

    # Faithfulness evaluation needs the exact context given to the LLM
    context = build_context(reranked_docs)

    return {
        "answer": answer,
        "context": context,
    }
class FaithfulnessScore(BaseModel):
    score: float = Field(
        ge=0.0,
        le=1.0,
        description="Faithfulness score between 0 and 1.",
    )
    reasoning: str = Field(
        description="Short explanation for the score.",
    )


FAITHFULNESS_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are evaluating the faithfulness of a RAG answer.

Determine whether the generated answer is supported by the retrieved context.

Scoring:
- 1.0 = Fully supported by the context.
- 0.5 = Partially supported; some claims are unsupported.
- 0.0 = Major claims are unsupported or contradict the context.

Do not judge whether the retrieved context itself is correct.
Do not use outside knowledge.
Judge only whether the answer is grounded in the provided context.
""".strip(),
    ),
    (
        "human",
        """
Question:
{question}

Retrieved Context:
{context}

Generated Answer:
{answer}
""".strip(),
    ),
])


def faithfulness_evaluator(run, example):
    question = example.inputs["question"]
    answer = run.outputs["answer"]
    context = run.outputs["context"]

    # Structured output returns the fields defined in FaithfulnessScore
    judge = get_llm().with_structured_output(FaithfulnessScore)
    chain = FAITHFULNESS_PROMPT | judge

    result = cast(FaithfulnessScore, chain.invoke({"question": question, "context": context, "answer": answer}))

    return {
        "key": "faithfulness",
        "score": result.score,
        "comment": result.reasoning,
    }

class AnswerRelevanceScore(BaseModel):
    score: float = Field(
        ge=0.0,
        le=1.0,
        description="Answer relevance score between 0 and 1.",
    )
    reasoning: str = Field(
        description="Short explanation for the score.",
    )


ANSWER_RELEVANCE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are evaluating the relevance of an AI-generated answer.

Determine how directly and completely the generated answer addresses
the user's question.

Scoring:
- 1.0 = Directly and completely answers the question.
- 0.5 = Partially answers the question or includes significant irrelevant information.
- 0.0 = Does not answer the question.

Do not judge factual correctness or faithfulness.
Judge only whether the answer is relevant to the question.
""".strip(),
    ),
    (
        "human",
        """
Question:
{question}

Generated Answer:
{answer}
""".strip(),
    ),
])


def answer_relevance_evaluator(run, example):
    question = example.inputs["question"]
    answer = run.outputs["answer"]

    # Judge only whether the generated answer addresses the question
    judge = get_llm().with_structured_output(AnswerRelevanceScore)
    chain = ANSWER_RELEVANCE_PROMPT | judge

    result = cast(AnswerRelevanceScore, chain.invoke({"question": question, "answer": answer}))

    return {
        "key": "answer_relevance",
        "score": result.score,
        "comment": result.reasoning,
    }

class AnswerCorrectnessScore(BaseModel):
    score: float = Field(
        ge=0.0,
        le=1.0,
        description="Answer correctness score between 0 and 1.",
    )
    reasoning: str = Field(
        description="Short explanation for the score.",
    )


ANSWER_CORRECTNESS_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are evaluating the correctness of an AI-generated answer.

Compare the generated answer with the reference answer and determine
whether the generated answer conveys the correct information.

Scoring:
- 1.0 = Fully correct and consistent with the reference answer.
- 0.5 = Partially correct but misses or misstates important information.
- 0.0 = Incorrect or contradicts the reference answer.

The generated answer does not need to use the exact same wording as
the reference answer.

Judge factual correctness, not writing style or verbosity.
""".strip(),
    ),
    (
        "human",
        """
Question:
{question}

Reference Answer:
{reference_answer}

Generated Answer:
{answer}
""".strip(),
    ),
])


def answer_correctness_evaluator(run, example):
    question = example.inputs["question"]
    answer = run.outputs["answer"]
    reference_answer = example.outputs["reference_answer"]

    # Compare the generated answer with our ground-truth answer
    judge = get_llm().with_structured_output(AnswerCorrectnessScore)
    chain = ANSWER_CORRECTNESS_PROMPT | judge

    result = cast(AnswerCorrectnessScore, chain.invoke({
        "question": question,
        "reference_answer": reference_answer,
        "answer": answer,
    }))

    return {
        "key": "answer_correctness",
        "score": result.score,
        "comment": result.reasoning,
    }

if __name__ == "__main__":
    # Evaluate generation quality across the LangSmith dataset
    client.evaluate(
        generation_target,
        data=DATASET_NAME,
        evaluators=[
            faithfulness_evaluator,
            answer_relevance_evaluator,
            answer_correctness_evaluator,
        ],
        experiment_prefix="generation-baseline",
        description="RAG generation evaluation for faithfulness, answer relevance, and answer correctness.",
    )