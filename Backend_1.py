

# !pip install vaderSentiment textblob
# !pip install emoji

# import nltk

# nltk.download("words")
# nltk.download('vader_lexicon')
# nltk.download('punkt')
# nltk.download('punkt_tab')

import pandas as pd
import re
import builtins
import emoji
import random
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from nltk.corpus import words
from nltk.tokenize import word_tokenize

v_analyzer = SentimentIntensityAnalyzer()

def testing_1():
      url = "https://raw.githubusercontent.com/Ankit152/IMDB-sentiment-analysis/master/IMDB-Dataset.csv"
      df = pd.read_csv(url)


      train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)
      print(f"Data Split Complete: {len(train_data)} training rows, {len(test_data)} test rows.\n")

      #  TEST DATA EVALUATION OF IMDB DATASET
      print("Running Evaluation on 10,000 Test Rows... please wait.")


      def get_hybrid_score(text):
          v_score = v_analyzer.polarity_scores(text)['compound']
          t_score = TextBlob(text).sentiment.polarity
          return (v_score + t_score) / 2


      test_data['final_score'] = test_data['review'].apply(get_hybrid_score)


      test_data['prediction'] = test_data['final_score'].apply(lambda x: 'positive' if x >= 0.05 else 'negative')


      final_acc = accuracy_score(test_data['sentiment'], test_data['prediction'])

      print("-" * 30)
      print(f"FINAL TEST ACCURACY: {final_acc*100:.2f}%")
      print("-" * 30)



      print(classification_report(test_data['sentiment'], test_data['prediction']))
      
def testing_2():
      url = "https://raw.githubusercontent.com/google-research/google-research/master/goemotions/data/train.tsv"
      df_2 = pd.read_csv(url, sep='\t', header=None, names=['text', 'labels', 'id'])

      emotion_list = [
          "admiration", "amusement", "anger", "annoyance", "approval", "caring",
          "confusion", "curiosity", "desire", "disappointment", "disapproval",
          "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
          "joy", "love", "nervousness", "optimism", "pride", "realization",
          "relief", "remorse", "sadness", "surprise", "neutral"
      ]

      positive_list = {
          "admiration", "amusement", "approval", "caring", "curiosity",
          "excitement", "gratitude", "joy", "love", "optimism", "pride", "relief"
      }

      negative_list = {
          "anger", "annoyance", "disappointment", "disapproval", "disgust",
          "embarrassment", "fear", "grief", "nervousness", "remorse", "sadness"
      }

      def categorize_binary(label_str):
          ids = [int(i) for i in str(label_str).split(',')]
          row_emotions = {emotion_list[i] for i in ids}


          is_positive = row_emotions.issubset(positive_list)

          is_negative = row_emotions.issubset(negative_list)

          if is_positive and not is_negative:
              return "positive"
          elif is_negative and not is_positive:
              return "negative"
          else:

              return None


      df_2['sentiment'] = df_2['labels'].apply(categorize_binary)


      binary_df = df_2.dropna(subset=['sentiment']).copy()

      train_data_2, test_data_2 = train_test_split(binary_df, test_size=0.5, random_state=42)
      print(f"Data Split Complete: {len(train_data_2)} training rows, {len(test_data_2)} test rows.\n")

      v_analyzer = SentimentIntensityAnalyzer()

      #  TEST DATA EVALUATION OF GoEmotion DATASET
      print("Running Evaluation on 12,000 Test Rows... please wait.")


      def get_hybrid_score(text):
          v_score = v_analyzer.polarity_scores(text)['compound']
          t_score = TextBlob(text).sentiment.polarity
          return (v_score + t_score) / 2


      test_data_2['final_score'] = test_data_2['text'].apply(get_hybrid_score)


      test_data_2['prediction'] = test_data_2['final_score'].apply(lambda x: 'positive' if x >= 0.05 else 'negative')


      final_acc_2 = accuracy_score(test_data_2['sentiment'], test_data_2['prediction'])

      print("-" * 30)
      print(f"FINAL TEST ACCURACY: {final_acc_2*100:.2f}%")
      print("-" * 30)



      print(classification_report(test_data_2['sentiment'], test_data_2['prediction']))

def get_bot_response(corrected_text):
    EXIT_COMMANDS = ["exit", "quit", "bye", "goodbye", "stop"]
    # Convert to lowercase and split into words for exact word matching
    words_in_text = word_tokenize(corrected_text.lower())

    if "hello" in words_in_text or "hi" in words_in_text:
        return " Hello!"
    if corrected_text in EXIT_COMMANDS:
        return " Goodbye!"
    v_score = v_analyzer.polarity_scores(corrected_text)['compound']
    t_score = TextBlob(corrected_text).sentiment.polarity
    final_score = (v_score + t_score) / 2

    if final_score >= 0.6:
         responses = ["WOW! That's amazing news! ", "I'm thrilled to hear that!", "That is wonderful!"]
         return f" Score ({final_score:.2f}): {random.choice(responses)}"

    elif 0.05 < final_score < 0.6:
       responses = ["That sounds great!", "I'm glad to hear that.", "Nice! "]
       return f" Score ({final_score:.2f}): {random.choice(responses)}"


    elif -0.05 <= final_score <= 0.05:
       return f" That's totally fine! Feel free to come back whenever you need anything. "


    elif -0.6 < final_score < -0.05:
      responses = ["I'm sorry to hear that.", "I understand, that's tough.", "I'm here to listen."]
      return f" Score ({final_score:.2f}): {random.choice(responses)}"

    elif final_score <= -0.6:
       responses = ["I can tell you're really upset. I'm so sorry.", "That sounds incredibly frustrating.", "I'm really sorry you're going through this."]
       return f" Score ({final_score:.2f}): {random.choice(responses)}"

def get_user_input():
  u_input = builtins.input("You: ").strip().lower()
  return u_input

def preprocess_text(user_input):
    user_input = user_input.lower()
    tokens = user_input.split()
    
    #english_words = set(words.words())
    words_in_text = word_tokenize(user_input.lower())
    # english_words = set(words.words())
    
    new_tokens=[]

    for word in tokens:
          # limit repeated punctuation (!!!! → !)
          word = re.sub(r"([!?.,])\1+", r"", word)
          
          if len(re.sub(r"(.)\1+", r"\1", word)) == 2 and word != "too":
                        
                # normalize repeated letters (soooo → so)
                word = re.sub(r"(.)\1+", r"\1", word)
                #print(f"{word}")
          else:
            
                # normalize repeated letters (happppy → happy)
                word = re.sub(r"(.)\1{2,}", r"\1\1", word)
                #print(f"{word}")
            
          new_tokens.append(word)

    tokens = new_tokens
    cleaned_text = " ".join(tokens)

    return cleaned_text

def empty_check(corrected_text):
  tokens = corrected_text.split()
  if len(tokens) == 0:
            return True
  return False

def typo_manage(user_input):
  user_input = user_input.lower()
  b = TextBlob(user_input)
  corrected_text = b.correct()
  return str(corrected_text)

def suggestion_validation(cleaned_text,corrected_text):
  if cleaned_text != corrected_text :
    d = input(f"Bot: Did you mean,{corrected_text} (Y/N):")
    #print(f"cleaned_text={cleaned_text},corrected_text={corrected_text}")
    if d.lower() == "y":
      corrected_text = corrected_text

    elif d.lower() == "n":
      corrected_text = cleaned_text

  return corrected_text

def gibberish_check(corrected_text):

  english_words = set(words.words())
  punctuations = [".",",","?","!"]
  informal_words = {
        "dont","don't","cant","can't","wont","won't",
        "gonna","wanna","gotta",
        "im","i'm","ive","i've","i'd",
        "youre","you're","theyre","they're","ok","okay","hi","hello"
    }
  tokens = corrected_text.split()

  #print(tokens)

  for word in tokens:

    if word.endswith("'s"):
       word = word[:-2]

    if word[len(word)-1] in punctuations and len(word) != 1:
      word = word[:-1]
      #print(word)

    if word not in  english_words and word not in  punctuations and word not in informal_words and not emoji.is_emoji(word) :
      return True

  return False

def start_interactive_session():
    print("--- Chatbot Active (Type 'bye' to stop) ---")
    EXIT_COMMANDS = ["exit", "quit", "bye", "goodbye", "stop"]
    while True:

          user_input = get_user_input()
          cleaned_text = preprocess_text(user_input)
          #print(f"{cleaned_text}")
          corrected_text = typo_manage(cleaned_text)
          #print(f"got it, {corrected_text}")
          corrected_text= suggestion_validation(cleaned_text,corrected_text)
          #print(f"got it, {corrected_text}")
          # if corrected_text in EXIT_COMMANDS:
          #   print("Bot: Goodbye!")
            # break

          if empty_check(corrected_text):
                print(" Please try somthing.")

          elif gibberish_check(corrected_text):
               print(" I'm sorry, I didn't quite catch that. Could you rephrase?")

          else:
               print(get_bot_response(corrected_text))
               if get_bot_response(corrected_text) == "goodbye":
                 exit

if __name__ == "__main__":
  testing_1()
  testing_2()
  start_interactive_session()


