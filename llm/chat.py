import CONFIGURATION
from ollama import chat
from chroma import relevantInformationRelatedTo

def generateResponse(messageHistory):
    lastMessage = messageHistory[-1].content
    additionalInformation = relevantInformationRelatedTo(lastMessage)
    additionalInformation = "\n".join(additionalInformation)

    return chat(
        model    = CONFIGURATION.CHAT_MODEL,
        messages =
        [
            {
                'role': 'system',
                'content': 
                    f"{CONFIGURATION.SYSTEM_PROMPT} only answer according to the following Information : {additionalInformation}. Say you dont know if no relevant information is provided"
            },
            *messageHistory
        ],
    ).message.content