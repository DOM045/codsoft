import re
import long_responses as long

# creaating a func that calculate probability
def message_probability(user_message, recognised_words, required_words=[]):
    message_certainty = sum(1 for word in user_message if word in recognised_words)
     #calculate percentage (accuracy %) of recognised owrd in user msg
    percentage = message_certainty / len(recognised_words)

    has_required_words = all(word in user_message for word in required_words)
# check taht the required words are in the string
    if has_required_words:
        return int(percentage * 100)
    else:
        return 0

def check_all_messages(message):
    highest_prob_list = {}

    def response(bot_response, list_of_words, required_words=[]):
        nonlocal highest_prob_list
        highest_prob_list[bot_response] = message_probability(message, list_of_words, required_words)
# responses-----------------------------------------------------------------------------------------
    response('Hello!', ['hello', 'hi', 'hey', 'sup', 'heyo'])
    response('See you!', ['bye', 'goodbye'])
    response('I\'m doing fine, and you?', ['how', 'are', 'you', 'doing'], required_words=['how'])
    response('You\'re welcome!', ['thank', 'thanks'])
    response('Thank you!', ['i', 'love', 'code', 'palace'], required_words=['code', 'palace'])
#long ans
    response(long.R_ADVICE, ['give', 'advice'], required_words=['advice'])
    response(long.R_EATING, ['what', 'you', 'eat'], required_words=['you', 'eat'])

    best_match = max(highest_prob_list, key=highest_prob_list.get)

    return long.unknown() if highest_prob_list[best_match] < 1 else best_match
#  the program continuously interacts with the user by asking for input.
def get_response(user_input):
    split_message = re.split(r'\s+|[,;?!.-]\s*', user_input.lower())
    response = check_all_messages(split_message)
    return response

while True:
    print('Bot: ' + get_response(input('You: ')))
