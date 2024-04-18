from assets.code.utils import pass_openAI_key
import openai
import json
import random

def generate_html_template(hero_title, hero_description, hero_images, features):
    selected_image = random.choice(hero_images)
    html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">
  <title>Monday.com Product Highlights</title>
</head>
<body class="bg-gray-100">
  <!-- Navigation -->
  <nav class="bg-white py-4">
    <div class="container mx-auto px-4 flex justify-between items-center">
      <a href="#">
        <img src="https://imagedelivery.net/XawdbiDo2zcR8LA99WkwZA/5757c917-86c3-4a4a-a3f7-4f9dff2dc300/public" alt="Monday.com Logo" class="h-8 w-auto">
      </a>
      <div>
        <a href="#features" class="text-gray-600 hover:text-gray-800 px-3 py-2">Features</a>
        <a href="#pricing" class="text-gray-600 hover:text-gray-800 px-3 py-2">Pricing</a>
        <a href="#" class="bg-black hover:bg-gray-800 text-white font-bold py-2 px-4 rounded">
          Sign Up
        </a>
      </div>
    </div>
  </nav>

  <!-- Hero Section -->
  <section class="bg-white py-20">
    <div class="container mx-auto px-4">
      <div class="flex flex-col md:flex-row items-center">
        <div class="md:w-1/2 mb-8 md:mb-0">
          <h1 class="text-4xl font-bold mb-4 text-gray-800">{hero_title}</h1>
          <p class="text-xl text-gray-600 mb-8">{hero_description}</p>
          <a href="#" class="bg-black hover:bg-gray-800 text-white font-bold py-3 px-6 rounded-lg">
            Start Free Trial
          </a>
        </div>
        <div class="md:w-1/2">
          <img src={selected_image} alt="Hero Image" class="w-full rounded-lg">
        </div>
      </div>
    </div>
  </section>

  <!-- Feature Sections -->
  <section class="py-20">
    <div class="container mx-auto px-4">
      {''.join(features)}
    </div>
  </section>

<!-- Pricing Section -->
 <section id="pricing" class="bg-white py-20">
    <div class="container mx-auto px-4">
      <h2 class="text-3xl font-bold mb-12 text-center text-gray-800">Pricing Plans</h2>
      <div class="grid grid-cols-1 md:grid-cols-3 gap-8">
        <!-- Pricing Plan 1 -->
        <div class="bg-gray-100 rounded-lg shadow-lg p-6">
          <h3 class="text-xl font-bold mb-4 text-gray-800">Basic</h3>
          <p class="text-4xl font-bold mb-4 text-gray-800">$9<span class="text-xl font-normal">/month</span></p>
          <ul class="text-gray-600 mb-8">
            <li class="mb-2">10 users included</li>
            <li class="mb-2">2 GB of storage</li>
            <li class="mb-2">Email support</li>
          </ul>
          <a href="#" class="bg-black hover:bg-gray-800 text-white font-bold py-2 px-4 rounded-lg block text-center">
            Get Started
          </a>
        </div>
        <!-- Pricing Plan 2 -->
        <div class="bg-gray-100 rounded-lg shadow-lg p-6">
          <h3 class="text-xl font-bold mb-4 text-gray-800">Pro</h3>
          <p class="text-4xl font-bold mb-4 text-gray-800">$29<span class="text-xl font-normal">/month</span></p>
          <ul class="text-gray-600 mb-8">
            <li class="mb-2">50 users included</li>
            <li class="mb-2">10 GB of storage</li>
            <li class="mb-2">Priority email support</li>
          </ul>
          <a href="#" class="bg-black hover:bg-gray-800 text-white font-bold py-2 px-4 rounded-lg block text-center">
            Get Started
          </a>
        </div>
        <!-- Pricing Plan 3 -->
        <div class="bg-gray-100 rounded-lg shadow-lg p-6">
          <h3 class="text-xl font-bold mb-4 text-gray-800">Enterprise</h3>
          <p class="text-4xl font-bold mb-4 text-gray-800">$99<span class="text-xl font-normal">/month</span></p>
          <ul class="text-gray-600 mb-8">
            <li class="mb-2">Unlimited users</li>
            <li class="mb-2">100 GB of storage</li>
            <li class="mb-2">Phone and email support</li>
          </ul>
          <a href="#" class="bg-black hover:bg-gray-800 text-white font-bold py-2 px-4 rounded-lg block text-center">
            Get Started
          </a>
        </div>
      </div>
    </div>
  </section>

  <!-- Testimonials Section -->
  <section class="bg-gray-100 py-20">
    <div class="container mx-auto px-4">
      <h2 class="text-3xl font-bold mb-12 text-center text-gray-800">What Our Customers Say</h2>
      <div class="grid grid-cols-1 md:grid-cols-3 gap-8">
        <!-- Testimonial 1 -->
        <div class="bg-white rounded-lg shadow-lg p-6">
          <p class="text-gray-600 mb-4">"Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nullam in dui mauris."</p>
          <div class="flex items-center">
            <img src="https://source.unsplash.com/random/80x80" alt="Customer 1" class="w-12 h-12 rounded-full mr-4">
            <div>
              <p class="font-bold text-gray-800">John Doe</p>
              <p class="text-gray-600">CEO, Company A</p>
            </div>
          </div>
        </div>
        <!-- Testimonial 2 -->
        <div class="bg-white rounded-lg shadow-lg p-6">
          <p class="text-gray-600 mb-4">"Vivamus luctus urna sed urna ultricies ac tempor dui sagittis."</p>
          <div class="flex items-center">
            <img src="https://source.unsplash.com/random/81x81" alt="Customer 2" class="w-12 h-12 rounded-full mr-4">
            <div>
              <p class="font-bold text-gray-800">Jane Smith</p>
              <p class="text-gray-600">CTO, Company B</p>
            </div>
          </div>
        </div>
        <!-- Testimonial 3 -->
        <div class="bg-white rounded-lg shadow-lg p-6">
          <p class="text-gray-600 mb-4">"Praesent ornare tortor ac ante egestas hendrerit."</p>
          <div class="flex items-center">
            <img src="https://source.unsplash.com/random/82x82" alt="Customer 3" class="w-12 h-12 rounded-full mr-4">
            <div>
              <p class="font-bold text-gray-800">Mark Johnson</p>
              <p class="text-gray-600">Founder, Company C</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  </section>

  <!-- Footer -->
  <footer class="bg-white py-8">
    <div class="container mx-auto px-4">
      <div class="flex flex-wrap justify-between">
        <div class="w-full md:w-1/3 mb-8 md:mb-0">
          <h3 class="text-xl font-bold mb-4 text-gray-800">MySaaS</h3>
          <p class="text-gray-600">123 Main Street<br>City, State 12345<br>Email: info@mysaas.com<br>Phone: (123) 456-7890</p>
        </div>
        <div class="w-full md:w-1/3 mb-8 md:mb-0">
          <h3 class="text-xl font-bold mb-4 text-gray-800">Quick Links</h3>
          <ul class="text-gray-600">
            <li class="mb-2"><a href="#" class="hover:text-gray-800">Home</a></li>
            <li class="mb-2"><a href="#features" class="hover:text-gray-800">Features</a></li>
            <li class="mb-2"><a href="#pricing" class="hover:text-gray-800">Pricing</a></li>
            <li class="mb-2"><a href="#" class="hover:text-gray-800">Contact Us</a></li>
          </ul>
        </div>
        <div class="w-full md:w-1/3">
          <h3 class="text-xl font-bold mb-4 text-gray-800">Follow Us</h3>
          <div class="flex">
            <a href="#" class="text-gray-600 hover:text-gray-800 mr-4"><i class="fab fa-facebook fa-lg"></i></a>
            <a href="#" class="text-gray-600 hover:text-gray-800 mr-4"><i class="fab fa-twitter fa-lg"></i></a>
            <a href="#" class="text-gray-600 hover:text-gray-800 mr-4"><i class="fab fa-instagram fa-lg"></i></a>
            <a href="#" class="text-gray-600 hover:text-gray-800"><i class="fab fa-linkedin fa-lg"></i></a>
          </div>
        </div>
      </div>
      <hr class="my-8 border-gray-300">
      <p class="text-center text-gray-600">&copy; 2023 MySaaS. All rights reserved.</p>
    </div>
  </footer>

</body>
</html>"""

    return html_template


def generate_feature_section(title, description, image_url):
    return f"""
      <div class="flex flex-col md:flex-row items-center mb-20">
        <div class="md:w-1/2">
          <img src="{image_url}" alt="{title} Image" class="w-full rounded-lg shadow-lg">
        </div>
        <div class="md:w-1/2 md:pl-8">
          <h2 class="text-3xl font-bold mb-4 text-gray-800">{title}</h2>
          <p class="text-lg text-gray-600 mb-8">{description}</p>
        </div>
      </div>
    """


def generate_content(recommendations, user_input = "", customer_name = "", customer_title = "", customer_company = "", model="gpt-3.5-turbo-0125"):
    # Get the OpenAI API key
    pass_openAI_key()

    # Extract the feature names and value propositions from the recommendations DataFrame into a list
    feature_list = recommendations["featureName"].tolist()
    value_prop_list = recommendations["valueProposition"].tolist()
    web_URl = recommendations["webURL"].tolist()

    # Create the conversation for the OpenAI API
    conversation = [
        {
            "role": "system",
            "content": "You are a helpful product marketing assistant designed to output JSON.",
        },
        {
            "role": "user",
            "content": f"""
            Here is the context of my conversation with the customer {customer_name}, {customer_title} from {customer_company}:
            {user_input}
            Based on their input, we have identified the following features and value propositions:
            Features:
            {feature_list}
            Value Propositions:
            {value_prop_list}

            Please generate a personalized title and a very short subtitle text for the hero section of the SaaS landing page. The response should be in JSON format with the following structure:
            {{
            "hero_title": "The title for the hero section of the SaaS landing page.",
            "hero_description": "A description for the hero section of the SaaS landing page."
            }}""",
        },
    ]

    # Generate the hero title and description using the OpenAI API
    response = openai.chat.completions.create(
        model=model,
        response_format={ "type": "json_object" },
        messages=conversation
    )
    response = json.loads(response.choices[0].message.content)
    hero_title = response["hero_title"]
    hero_description = response["hero_description"]


    # Initialize empty lists to store the generated feature titles and value propositions
    feature_titles = []
    value_propositions = []

    # Iterate over all the entries
    for i in range(len(feature_list)):
        # Create the conversation for the OpenAI API
        conversation2 = [
            {
                "role": "system",
                "content": "You are a helpful product marketing assistant designed to output JSON.",
            },
            {
                "role": "user",
                "content": f"""
                Here is the context of my conversation with the customer {customer_name}, {customer_title} from {customer_company}:
                {user_input}
                
                Please generate a personalized title and a very short subtitle text to highlight the feature {feature_list[i]} and value prop {value_prop_list[i]} for the customer on a landing page. The response should be in JSON format with the following structure:
                {{
                    "feature_title": "The title for the feature section of the SaaS landing page.",
                    "value_proposition": "A value proposition for the feature section of the SaaS landing page."
                }}""",
            },
        ]

        # Generate the feature title and value proposition using the OpenAI API
        response = openai.chat.completions.create(
            model=model,
            response_format={
                "type": "json_object"
            },
            messages=conversation2
        )

        response = json.loads(response.choices[0].message.content)
        feature_title = response["feature_title"]
        value_proposition = response["value_proposition"]

        # Append the generated feature title and value proposition to the respective lists
        feature_titles.append(feature_title)
        value_propositions.append(value_proposition)

    return hero_title, hero_description, feature_titles, value_propositions, web_URl
    
