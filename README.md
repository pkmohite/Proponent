<p align="left">
  <img src="assets/images/logo_full_black.png" alt="Proponent Logo" width="300">
</p>

<h1 align="left">Introducing Proponent: Empowering Product-Led Sales with AI</h1>

<p align="left">
  <a href="#features">Features</a> •
  <a href="#key-differentiation">Key Differentiation</a> •
  <a href="#project-status">Project Status</a> •
  <a href="#contributing">Contributing</a> •
  <a href="#license">License</a> •
  <a href="#contact">Contact</a>
</p>

---

Proponent is an NLU-powered sales enablement tool designed to help product-led businesses drive revenue growth. By leveraging AI and customer insights, Proponent generates hyper-relevant content and enables sales teams to have more effective conversations with prospects.


## Features

- **Intelligent Content Generation**: AI-powered creation of persuasive, tailored content for each prospect
- **Actionable Customer Insights**: Analyzes customer data to surface insights for personalized outreach
- **Messaging Manager**: Enables easy management of product messaging and sales enablement materials
- **Customer Personas**: Gathers customer profile data to refine messaging strategies and targeting


## Getting Started 
To run the Proponent application locally:

1. Clone the repository:
  ```
  git clone https://github.com/yourusername/proponent.git
  ```

2. Install dependencies:
  ```
  poetry install
  ```

3. Run the Streamlit app:
  ```
  poetry run streamlit run app.py
  ```

Alternatively, you can access the deployed Proponent application on Streamlit Sharing at [https://proponent.streamlit.app/](https://proponent.streamlit.app/).


## Contributing
We welcome contributions to help make Proponent even better. If you'd like to contribute, please follow these steps:

Fork the repository
Create a new branch: git checkout -b feature/your-feature
Make your changes and commit them: git commit -m 'Add your feature'
Push to the branch: git push origin feature/your-feature
Open a pull request
Please note that for feature contributions, we only accept features built using Streamlit native elements, components, and experimental features. We are not open to non-official elements at this time.

Please note that Proponent is currently under active development and may not be fully ready for deployment yet. We are working diligently to refine the features, improve the codebase, and ensure a seamless user experience. Stay tuned for updates on the project's progress and future release plans.

## License

This project is licensed under the [Apache License 2.0](LICENSE).

## Tech Stack

- Streamlit for the web application framework
- Poetry for dependency management and packaging
- Python for the core programming language
- OpenAI API for NLU & AI-powered content generation
- Pandas for data manipulation and analysis
- Parquet for storage & logging
- NumPy for numerical computing
- Matplotli for data visualization

proponent/
│
├── assets/
│   └── images/
│       └── logo_full_black.png
│
├── pages/
│   ├── Customer_Analytics.py
│   └── Messaging_Manager.py
│
├── templates/
│   ├── customer_personas.csv
│   ├── messaging_framework.csv
│   └── sales_playbook.md
│
├── .gitignore
├── LICENSE
├── README.md
├── app.py
├── poetry.lock
└── pyproject.toml


## Contact
For more information about Proponent, visit our <a href="https://www.yourproponent.com" style="color: #007A6D; text-decoration: none;">website</a>.

If you have any questions, suggestions, or feedback, feel free to reach out:

Prashant Mohite (Founder)
<a href="mailto:prashant@yourproponent.com" style="color: #007A6D; text-decoration: none;">Email</a>
<a href="https://www.linkedin.com/in/prashantmohite/" style="color: #007A6D; text-decoration: none;">LinkedIn</a>