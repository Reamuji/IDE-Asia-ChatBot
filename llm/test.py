# # print(pow(1.007,12))
# # print(1000000*1/0.0052616942768478)
# # print(2000000*1/0.0052616942768478)
# # print(3000000*1/0.0052616942768478)
# # print(4000000*1/0.0052616942768478)
# print(5000000*1/0.0052616942768478)

# from ollama import chat
# from chromaDB import relevantInformationAbout

from chat import generateResponse

# print("\n".join(relevantInformationAbout("kontak")))


messageHistory = []
while True:
    user_input = input('User: ')

    messageHistory += [
        {'role': 'user', 'content': user_input}
    ]
    
    botResponse = generateResponse(messageHistory)
    print('AI: '+ botResponse)
    
    messageHistory += [
        {'role': 'assistant', 'content': botResponse},
    ]