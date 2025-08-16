from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field


class GradeDocuments(BaseModel):
    """A model for grading documents based on their relevance to a question."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


def create_document_grader(llm):
    """Creates a document grader that uses a language model to determine the relevance of documents to a user's question.

    Args:
        llm: A language model instance that supports structured output

    Returns:
        A grader chain that takes a dictionary with 'document' and 'question' keys
        and returns a GradeDocuments object with a binary_score ('yes' or 'no')
        indicating whether the document is relevant to the question
    """
    # Define the structured output model for grading documents
    structured_llm_documents_grader = llm.with_structured_output(GradeDocuments)

    documents_grader_system_prompt = """You are a document relevance grader for cybersecurity questions.
Your task: Determine if a document contains information relevant to answering the user's question.
You MUST respond with exactly "yes" or "no" in the binary_score field."""

    # Define the prompt template for grading documents
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
