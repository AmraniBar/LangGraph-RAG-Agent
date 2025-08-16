from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain import hub


def create_rag_chain(llm):
    """Creates a RAG chain for answering questions based on retrieved context.

    Args:
        llm: A language model instance for generating answers

    Returns:
        A chain that takes a dictionary with 'question' and 'context' keys
        and returns a string answer based on the provided context
    """

    rag_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "human",
                """Based on the following context, answer this question: {question}

Context: {context}

Answer:""",
            )
        ]
    )

    return rag_prompt | llm | StrOutputParser()
