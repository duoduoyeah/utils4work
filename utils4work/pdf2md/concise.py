from pyzerox import zerox
import os
import json
import asyncio
import dotenv

dotenv.load_dotenv()

### Model Setup (Use only Vision Models) Refer: https://docs.litellm.ai/docs/providers ###

## placeholder for additional model kwargs which might be required for some models
kwargs = {}

## system prompt to use for the vision model
custom_system_prompt = None

# to override
# custom_system_prompt = "For the below PDF page, do something..something..." ## example

###################### Example for OpenAI ######################
model = "gpt-4o-mini"  ## openai model
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")  ## your-api-key

if False:
    ###################### Example for Azure OpenAI ######################
    model = (
        "azure/gpt-4o-mini"  ## "azure/<your_deployment_name>" -> format <provider>/<model>
    )
    os.environ["AZURE_API_KEY"] = ""  # "your-azure-api-key"
    os.environ["AZURE_API_BASE"] = ""  # "https://example-endpoint.openai.azure.com"
    os.environ["AZURE_API_VERSION"] = ""  # "2023-05-15"


    ###################### Example for Gemini ######################
    model = "gemini/gpt-4o-mini"  ## "gemini/<gemini_model>" -> format <provider>/<model>
    os.environ["GEMINI_API_KEY"] = ""  # your-gemini-api-key


    ###################### Example for Anthropic ######################
    model = "claude-3-opus-20240229"
    os.environ["ANTHROPIC_API_KEY"] = ""  # your-anthropic-api-key

    ###################### Vertex ai ######################
    model = "vertex_ai/gemini-1.5-flash-001"  ## "vertex_ai/<model_name>" -> format <provider>/<model>
    ## GET CREDENTIALS
    ## RUN ##
    # !gcloud auth application-default login - run this to add vertex credentials to your env
    ## OR ##
    file_path = "path/to/vertex_ai_service_account.json"

    # Load the JSON file
    with open(file_path, "r") as file:
        vertex_credentials = json.load(file)

    # Convert to JSON string
    vertex_credentials_json = json.dumps(vertex_credentials)

    vertex_credentials = vertex_credentials_json

    ## extra args
    kwargs = {"vertex_credentials": vertex_credentials}

###################### For other providers refer: https://docs.litellm.ai/docs/providers ######################


# Define main async entrypoint
async def main():
    file_path = "/workspaces/simulation/paper/0312_tests/daisen/gme.pdf"  ## local filepath and file URL supported

    ## process only some pages or all
    select_pages = list(range(1, 13))  ## Select pages 1 through 12 (inclusive)

    output_dir = "./output/md_from_pdf"  ## directory to save the consolidated markdown file
    result = await zerox(
        file_path=file_path,
        model=model,
        output_dir=output_dir,
        custom_system_prompt=custom_system_prompt,
        select_pages=select_pages,
        **kwargs
    )
    return result


# run the main function:
result = asyncio.run(main())

# print markdown result
print(result)
