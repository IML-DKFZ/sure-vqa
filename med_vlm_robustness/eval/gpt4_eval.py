import json
from openai import OpenAI
import os
import ndjson

OPEN_AI_API_KEY = os.getenv("OPEN_AI_API_KEY")

def gpt4_eval():

    # get the path to the current file
    path_of_this_script = os.path.dirname(__file__)

    # read the initial prompt to guide gpt4
    with open(
                path_of_this_script + '/../gpt4_instructions.txt',
                'r'
        ) as f:
        initial_prompt = f.read()

    # load the json file containing the data we wanna evaluate
    with open(path_of_this_script + '/../gpt4_input_data.json', 'r') as file:
        # Load the JSON data
        json_file = file.read()
    input_data = json.loads(json_file)

    # get access to gpt4 api
    client = OpenAI(api_key=OPEN_AI_API_KEY)
    output = []
    full_output_log = []

    # Iterate over each item in the JSON array
    for line in input_data:
        response = client.chat.completions.create(
                        model="gpt-4",
                        messages=[
                            {
                                "role": "user",
                                "content": initial_prompt
                            },
                            {
                                "role": "assistant",
                                "content": "This is clear, and I have no questions. I am ready to start."
                            },
                            {
                                "role": "user",
                                "content": str(line)
                            }
                        ]
                    )
        print(response)
        output.append(json.dumps(response.choices[0].message.content))
        # full_output_log.append(response)

    # we can use this code to store the complete output of gpt4 
    # TODO: Needs a bug fix 
    # # write to files
    # with open(f'/home/localuser/Desktop/projects/med_vlm_robustness/med_vlm_robustness/final_output.json', 'w') as f:
    #     ndjson.dump(full_output_log, f)
        
    # save the results to a json file
    with open(path_of_this_script + '/../gpt4_eval_output.json', 'w') as f:
        ndjson.dump(output, f)

gpt4_eval()