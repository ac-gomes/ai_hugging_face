
def output_formatter(result: dict) -> str:
    """ Format the output received from  🤗 transformers pipeline """

    try:
        formatted_output = "{\n"

        if len(result) > 0:

            for key, value in result.items():

                formatted_output += f"{chr(32)*3}'{key}': {value},\n"

            formatted_output += "}"

            print(formatted_output)

    except Exception as err:
        print(f"Something went wrong, \n error: {err}")


def edecoded_sequence(result: str) -> str:

    try:
        if result is not None:

            print(f"\n >> decoded_sentence: {result} \n")

    except Exception as err:
        print(f"Something went wrong, \n error: {err}")


def pre_tokenizer_output_formatter(result) -> str:

    try:
        formatted_output = ""

        if result is not None:
            for item in result:

                formatted_output += f'"{item[0]}", '

        print(f'\n >> resultado: {formatted_output}\n')

    except Exception as err:
        print(f"Something went wrong, \n error: {err}")
