# smart-meter
Domestic energy experiments involving EDF and smart meters.  Note that in order to use these scripts, in addition to having an EDF account, you will need to have a valid OpenAI API key set in an `OPEN_API_KEY` environment variable.  You will also need to know your [MPAN number](https://en.wikipedia.org/wiki/Meter_Point_Administration_Number) and [IHD number](https://www.equiwatt.com/help/where-do-i-find-the-mac/guid/eui-number-on-my-in-home-display-ihd) and you will need to set environment variables for them both.

* **[smart-meter-playbook](smart-meter-playbook.ipynb)** - a Jupyter notebook walking through how to access your smart meter data from the [DCC](https://www.smartdcc.co.uk/about-dcc/) using your supply point [MPAN number](https://en.wikipedia.org/wiki/Meter_Point_Administration_Number) and your [IHD number](https://www.equiwatt.com/help/where-do-i-find-the-mac/guid/eui-number-on-my-in-home-display-ihd).
 
* **[edf-bill-playbook](edf-bill-playbook.ipynb)** - a Jupyter notebook walking through how to process the energy data you can download from the EDF customer portal as shown below:
<img width="653" alt="image" src="https://github.com/user-attachments/assets/5556df8f-387f-4b03-8007-ebbf8429c212">

* **[energy-advisor.py](energy-advisor.py)** - a command line agent entirely developed using ChatGPT Pro and Cursor which is able to autogenerate an HTML insights report plus graph from that EDF csv. The report looks like this:
<img width="304" alt="image" src="https://github.com/user-attachments/assets/accb45ce-ce44-4ed1-ab5c-c3649018ca3d">


