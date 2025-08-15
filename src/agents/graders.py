from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field


class GradeDocuments(BaseModel):
    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


# class GradeHallucinations(BaseModel):
#     binary_score: str = Field(
#         description="Answer is grounded in the facts, 'yes' or 'no'"
#     )


# class GradeAnswer(BaseModel):
#     binary_score: str = Field(
#         description="Answer addresses the question, 'yes' or 'no'"
#     )


def create_document_grader(llm):
    structured_llm_documents_grader = llm.with_structured_output(GradeDocuments)

    documents_grader_system_prompt = """You are a document relevance grader for cybersecurity questions.
Your task: Determine if a document contains information relevant to answering the user's question.
You MUST respond with exactly "yes" or "no" in the binary_score field."""

    # CRITICAL: You must return a structured response with ONLY "yes" or "no" in the binary_score field.

    # Rules:
    # - If the document contains any information that could help answer the question, return "yes"
    # - If the document does not contain helpful information, return "no"
    # - Look for relevant keywords, concepts, or topics
    # - For cybersecurity questions, frameworks, standards, controls, and procedures are typically relevant
    # - Be more permissive rather than restrictive
    documents_grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", documents_grader_system_prompt),
            (
                "human",
                "Retrieved document: \\n\\n {document} \\n\\n User question: {question}",
            ),
        ]
    )
    return documents_grade_prompt | structured_llm_documents_grader


# def create_hallucination_grader(llm):
#     structured_llm_hallucination_grader = llm.with_structured_output(
#         GradeHallucinations
#     )

#     hallucination_grader_system_prompt = """You are a grader checking if an LLM generation is grounded in or supported by a set of retrieved facts.
# Give a simple 'yes' or 'no' answer. 'Yes' means the generation is grounded in or supported by a set of retrieved the facts."""

#     hallucination_grader_prompt = ChatPromptTemplate.from_messages(
#         [
#             ("system", hallucination_grader_system_prompt),
#             (
#                 "human",
#                 "Set of facts: \\n\\n {documents} \\n\\n LLM generation: {generation}",
#             ),
#         ]
#     )
#     return hallucination_grader_prompt | structured_llm_hallucination_grader


# def create_answer_grader(llm):
#     answer_structured_llm_grader = llm.with_structured_output(GradeAnswer)

#     answer_grader_system = """You are a grader assessing whether an answer addresses / resolves a question \\n

#      Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""

#     answer_prompt = ChatPromptTemplate.from_messages(
#         [
#             ("system", answer_grader_system),
#             (
#                 "human",
#                 "User question: \\n\\n {question} \\n\\n LLM generation: {generation}",
#             ),
#         ]
#     )

#     return answer_prompt | answer_structured_llm_grader
