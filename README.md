# GeneFit-Backend
Backend Setup for the Genefit Project 

What is GeneFit ? 

Genetic Data is an unexplored treasure , which when delved deep into can prove to be the solution to a lot of problems. GeneFit utilises the goodness of the genetic Data with the basic medical information (age, gender, weight , height , activity level , medical history ) and provides suggestive lifestyle  advices . Not Only That it will utilise the provided genetic data and warn user about the possibilty of the occurence of any disease in the future so that the user can adapt to the preventive measures ASAP!

The current implemented application Flow begins with the User Details Input Page where the user is required to enter the information as stated for an individual . 
The information required are listed as follows : 
Name , Age , Gender , Weight , Height , Diagnosed History , Activity Level (Basic , Moderate , Intense) 
and an upload field for the csv file that consists of the indivual genetic data. 

Based on the above input fields except the genetic data file lifestyle advice is generated in accordance with the following criteria : 
Health Status 
Recommeded Calorie Intake
Recommended Protein Intake
Other MacroNutrient Intake 
General Life Style Advice with the medical history taken into account

This Repository provides with the backend Setup for GeneFit. 

The source folder consists of the "genetic_sequence_predictor.keras" file which is the model that will predict the diseases based on the uploaded genetic data. The model has been trained on the mapped data between the FastQ records from the "[1000 Genomes Project]([url](https://www.internationalgenome.org/)) " and the "variant_summary.txt" from [ClinVAR](https://www.ncbi.nlm.nih.gov/clinvar/). 

Model details for the enthusiasts: :)

Model Accuracy :
![image](https://github.com/user-attachments/assets/1b531687-665c-4938-beff-2729335ebe5a)

Plots: 

![image](https://github.com/user-attachments/assets/2cf5c006-8a23-4327-9a5c-3aff0fb8ffe7)



In the Current Implementation the General LifeStyle Advice (NOT the disease prediction for the future based on genetic Data) is generated by utilising a prompt Engineered Google Gemini Flash Model . This will be improved upon in the future iterations. 
For setting up the Backend onto your local machine follow the below steps: 

```**Clone the repo onto your localmachine .** ```

```**Install the relvant Python pip libraries **```

```**Run api2.py\```





GeneFit is paired with an amazing FrontEnd Application with modern UI . You can check the FrontEnd Application Repository [here](https://github.com/vishalverma9572/medecro_frontend)




