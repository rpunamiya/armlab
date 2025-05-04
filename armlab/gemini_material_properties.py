import google.generativeai as genai
import os

class GeminiClass:
    def __init__(self, prompt=None):
        """
        Constructor for the GeminiClass.
        Initializes the generative model and sets a default prompt.

        Parameters:
        - prompt (optional): A string to set as the default prompt. Defaults to an empty string if not provided.
        """
        self.gemini_model = genai.GenerativeModel("gemini-1.5-flash")
        if prompt:
            self.prompt = prompt
        else:
            self.prompt = ""

    def set_prompt(self, prompt):
        """
        Sets or updates the prompt for the GeminiClass.

        Parameters:
        - prompt: A string that will serve as the base prompt for content generation.
        """
        #Add code to set self.prompt to the provided prompt
        self.prompt = prompt

    def generate_content(self, text, usePrompt):
        """
        Generates content using the generative model.

        Parameters:
        - text: A string input to provide to the model for content generation.
        - usePrompt: A boolean indicating whether to prepend the class's prompt to the input text.

        Returns:
        - The generated content as a string.
        """
        #Add code to generate content from text. The usePrompt boolean should determine if the prompt is used or not.
        if usePrompt:
            text = self.prompt + text
        generated_content = self.gemini_model.generate_content(text)
        return generated_content

    def generate_from_image(self, image_bytes, textInput):
        """
        Generates content based on an image and optional text input.

        Parameters:
        - image_bytes: The binary data of the image.
        - textInput: A string input to accompany the image for content generation.

        Returns:
        - The generated content as a string.
        """
        # Add code that allows for the image and the text input to be passed to Gemini and return Gemini's response.
        #Hint: Gemini can accept multiple inputs in the form of a list.
        generated_content = self.gemini_model.generate_content([image_bytes, textInput]).text
        return generated_content

# Example usage of the GeminiClass
if __name__ == "__main__":
    #Use this to test out the API
    json_key_path = r"C:\Users\rpuna\OneDrive - Stanford\Research\ARMLab\armlab\me326-hw2-02b887b1a25c.json"
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = json_key_path

    gemini = GeminiClass()

    # Gemini should provide a list of average material properties of a list of materials (variable name: materials)
    materials = ["aluminum", "steel", "wood", "concrete", "paper"]
    gemini.set_prompt("Please provide a list of the average yield strength properties for these materials. Your response should be a list of numbers, without units (assume MPa).")
    content = gemini.generate_content(str(materials), True).text
    # convert the string to a list of floats
    yields = [float(x.replace(',', '')) for x in content.split()]
    print(yields)
    smallest = min(yields)
    print("Smallest Yield Strength:", smallest, "MPa")
    print("Material with smallest yield strength:", materials[yields.index(smallest)])
