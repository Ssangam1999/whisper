import spacy
import cv2
import json
import uuid
from num2words import num2words
from datetime import datetime

from whisper.transcribe import cli



# Function to extract entities from text
def extract_entities(text, nlp_model):
    """
    Takes text and nlp_model as input.
    Process it to get person money and organisation
    returns persons,money,orgs
    """
    doc = nlp_model(text)
    persons = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
    money = [ent.text for ent in doc.ents  if ent.label_ == "MONEY" ]
    orgs = [ent.text for ent in doc.ents if ent.label_ == "ORG"  ]
    return persons, money, orgs

# Function to save NER output to a JSON file
def save_ner_to_json(persons, money, orgs, output_path):
    """
    Takes the output of extract_entities
    Writes them in json file and dump it
    """
    # Create a dictionary to hold the data
    ner_output = {
        "person": persons[0] if persons else "Unknown",
        "amount": money[0] if money else "Unknown",
        "organization": orgs[0] if orgs else "Unknown"
    }

    # Write the dictionary to a JSON file
    with open(output_path, 'w') as json_file:
        json.dump(ner_output, json_file, indent=4)

    print(f"NER output saved to {output_path}")


# Function to fill the check image
def fill_check(image_path, output_path, positions, name, amount, amount_words, date):
    """
    image_path = path of an input image
    output_path = path of output image
    position = Where to write information in check
    name = name of the person in check
    amount =  amount to be written in check
    amount_words = amount in words
    date = date in which check is issued
    """

    check_image = cv2.imread(image_path)

    # Define font properties
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_thickness = 2
    color = (0, 0, 0)

    # Write details to the check
    cv2.putText(check_image, name, positions['name'], font, font_scale, color, font_thickness)
    cv2.putText(check_image, amount, positions['amount'], font, font_scale, color, font_thickness)
    cv2.putText(check_image, amount_words, positions['amount_words'], font, font_scale, color, font_thickness)
    cv2.putText(check_image, date, positions['date'], font, font_scale, color, font_thickness)

    # Save and display the updated image
    cv2.imwrite(output_path, check_image)
    cv2.imshow("Filled Check", check_image)
    cv2.waitKey(0)
    print(f"Check saved to {output_path}")


# Main processing function
def process_check(persons, money, orgs):
    """
    Process check to write information on respective location
    persons =  name of person to whom check is written
    money= amount of money in check
    orgs = organisation through which check has been issued
    """
    if not orgs or not orgs[0]:
        print("We don't support this bank currently.")
        return

    bank_positions = {
        "Bank of America": {
            "image_path": "/home/vertex/Documents/vertex_projects/whisper/data/bank of america.png",
            "positions": {
                "name": (80, 80),
                "amount": (400, 80),
                "amount_words": (20, 100),
                "date": (220, 50)
            }
        },
        "Chase Bank": {
            "image_path": "/home/vertex/Documents/vertex_projects/whisper/data/chasebank.jpeg",
            "positions": {
                "name": (120, 130),
                "amount": (580, 140),
                "amount_words": (50, 170),
                "date": (470, 90)
            }
        },
        "U.S. Bank": {
            "image_path": "/home/vertex/Documents/vertex_projects/whisper/data/usbank.jpeg",
            "positions": {
                "name": (120, 130),
                "amount": (580, 140),
                "amount_words": (50, 170),
                "date": (470, 90)
            }
        }
    }

    bank_name = orgs[0]
    if bank_name not in bank_positions:
        print("We don't support this bank.")
        return

    details = bank_positions[bank_name]
    output_path = "filled_check_opencv.png"
    name = persons[0]
    amount = money[0]
    amount_words = num2words(amount)
    present_date = str(datetime.now().date())

    fill_check(details["image_path"], output_path, details["positions"], name, amount, amount_words, present_date)


if __name__ == "__main__":
    nlp = spacy.load("en_core_web_trf")
    text = cli(['/home/vertex/Documents/vertex_projects/whisper/CHASE.mp3'])
    print("Transcribed Text:", text)

    # # # Extract entities
    # persons, money, orgs = extract_entities(text, nlp)
    # # # Print extracted entities
    # print("Persons:", persons)
    # print("Money:", money)
    # print("Organizations:", orgs)
    # output_path = f'/home/vertex/Documents/vertex_projects/whisper/whisper/output/{uuid.uuid4()}.json'
    #
    # #save json
    # save_ner_to_json(persons, money, orgs, output_path)


    # Process the check
    # process_check(persons, money, orgs)

