---
layout: archive
title: "CV"
permalink: /cv/
author_profile: true
redirect_from:
  - /resume
---

{% include base_path %}

Education
======
* M.S., Data Science	| Seattle University (_Sep 2024 - May 2025_)	 			        		
* B.E., Information Technology | Muffakham Jah college of Engineering and Technology (_2017 - 2021_)


Work experience
======

**Graduate Research Assistant @ Seattle University (_June 2024 - Present_)**
* Working on neuroscience research project on barn owls, where I analyze neurophysiological data that contains neural responses to different auditory stimuli, with the goal of understanding how these owls localize sound with such precision.
* Extracting data from published figures, re-creating them, organizing the data from neurobiological experiments to ensure seamless data analysis.
* Applying various regression techniques to fit the data and plot a correlation between different responses of over 120 different neurons.
* Presenting a poster at the Neuroscience Conference 2024 to showcase the latest findings of our analysis. 

**Software/Data Engineer @ Applied Information Sciences (_Aug 2021 – Aug 2023_)**
* Worked on building an end-to-end Data Transformation ELT Solution that handled **millions of records** of a leading auto insurance company in the USA.
* Built an Audit Framework to validate the data processed by the ELT solution.
* Developed a separate Test framework of around **(~60 unit tests)** to perform Unit Testing on the complete ELT solution.
* Built a PDF Q&A chat application using OpenAI API, Large Language Models & Semantic Kernels SDK.
* Integrated vector database Qdrant with LLMs to store vectorized data and fetch most relevant responses to the query.
* Worked on Prompt Engineering best practices and utilized them to enhance the query efficiency.

Skills
======
* Skill 1
* Skill 2
  * Sub-skill 2.1
  * Sub-skill 2.2
  * Sub-skill 2.3
* Skill 3

Publications
======
  <ul>{% for post in site.publications reversed %}
    {% include archive-single-cv.html %}
  {% endfor %}</ul>
  
Talks
======
  <ul>{% for post in site.talks reversed %}
    {% include archive-single-talk-cv.html  %}
  {% endfor %}</ul>
  
Teaching
======
  <ul>{% for post in site.teaching reversed %}
    {% include archive-single-cv.html %}
  {% endfor %}</ul>
  
Service and leadership
======
* Currently signed in to 43 different slack teams
