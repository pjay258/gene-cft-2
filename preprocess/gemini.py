import google.generativeai as genai
import google.ai.generativelanguage as glm
import pathlib

# 예외 처리 필요함, 가끔 에러 발생 try 사용할 것
def generate_instructions(target_obj: str, image_PATH: str, api_key: str):
    genai.configure(api_key=api_key)
    
    model = genai.GenerativeModel('gemini-pro-vision')
    response = model.generate_content(
        glm.Content(
            parts = [
                glm.Part(text=\
                        f'''Devise a set of instructions aimed at modifying diverse attributes of the object and background in the provided image.

                        For example, when confronted with an ocean image, you could direct to "water: transform water into lava," "waves: eliminate waves," or "surfers: introduce additional surfers." Likewise, in a portrait of a smiling man, contemplate instructions such as "face: evoke a crying expression," "hat: place a hat on his head," "eyewear: transition glasses into sunglasses," "act: depict him in a sitting position," or "texture: morph him into stone."

                        Attributes include backgrounds, actions, situations, facial expressions, tools, features, etc. Formulate a unique instruct for each attribute, ensuring a diverse selection.

                        Identify significant and distinct keywords (beyond background) and generate a varied set of coherent and reasonable instructs. The quantity of keywords can be adapted based on the complexity of the image.

                        Present the instructs in the format below:

                        * keyword1: detailed instruct
                        * keyword2: detailed instruct
                        * keyword3: detailed instruct
                        * keyword4: detailed instruct
                        * keyword5: detailed instruct
                        * keyword6: detailed instruct
                        * keyword7: detailed instruct
                        * keyword8: detailed instruct
                        * keyword9: detailed instruct
                        * keyword10: detailed instruct
                        ...

                        Ensure that the {target_obj} remains unchanged.
                        '''),
                glm.Part(
                    inline_data=glm.Blob(
                        mime_type='image/png',
                        data=pathlib.Path(image_PATH).read_bytes()
                    )
                ),
            ],
        ),
        stream=True)
    response.resolve()

    return response.text


# 예외 처리 필요함, 가끔 에러 발생 try 사용할 것
def generate_description(model, target_obj: str, image_PATH: str, api_key: str):
    response = model.generate_content(
        glm.Content(
            parts = [
                glm.Part(text=\
                        f'''Descibe this {target_obj} in a single short sentence. For examples, "a white cat", "a huge house", "a smiling man" '''),
                glm.Part(
                    inline_data=glm.Blob(
                        mime_type='image/png',
                        data=pathlib.Path(image_PATH).read_bytes()
                    )
                ),
            ],
        ),
        stream=True)
    response.resolve()

    return response.text