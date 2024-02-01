## Table of Contents
- [Openai API key](#openai-api-key)

## Setting up the workspace
<a name="openai-API-key"></a>
## Openai API key
To enable the gpt4 evaluation functionality in this repo you need to have an API key from Openai.

To get an API key please go to the following webpage: https://platform.openai.com/docs/overview

After getting a key, you need to add this key to the .env file in your repository. The file, *template.env*, is created for you as a guide which shows how your .env file should look like. You can copy the context of this file to your .env file or basically change the file name to .env by removing the template part. Note that the repository will not run properly if you don't set all the parameters spesified in this file.

You can add your API key to the following variable in the .env file:

```shell
export OPEN_AI_API_KEY= your_api_key
```

Once you have set your key you need to run the following before starting the program.

```shell
source .env 
```

## Setting up OVQA Dataset
- change the train, val and test set names as test.json train.json validate.json
