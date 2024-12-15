# smart-meter
Domestic energy experiments involving EDF and smart meters.  Note that in order to use these scripts, in addition to having an EDF account, you will need to have a valid OpenAI API key set in an `OPEN_API_KEY` environment variable.  You will also need to know your [MPAN number](https://en.wikipedia.org/wiki/Meter_Point_Administration_Number) and [IHD number](https://www.equiwatt.com/help/where-do-i-find-the-mac/guid/eui-number-on-my-in-home-display-ihd) and you will need to set environment variables for them both.

* **[smart-meter-playbook](smart-meter-playbook.ipynb)** - a Jupyter notebook walking through how to access your smart meter data from the [DCC](https://www.smartdcc.co.uk/about-dcc/) using your supply point [MPAN number](https://en.wikipedia.org/wiki/Meter_Point_Administration_Number) and your [IHD number](https://www.equiwatt.com/help/where-do-i-find-the-mac/guid/eui-number-on-my-in-home-display-ihd).
 
* **[edf-bill-playbook](edf-bill-playbook.ipynb)** - a Jupyter notebook walking through how to process the energy data you can download from the EDF customer portal per the screenshot below.  The csv which has rows corresponding to months and five columns as follows:
```
Timestamp,Electricity consumption (kWh),Electricity cost (£),Gas consumption (kWh),Gas cost (£)
08/2024,38.788,27.39,135.1189,19.95
``` 
<img width="653" alt="image" src="https://github.com/user-attachments/assets/5556df8f-387f-4b03-8007-ebbf8429c212">

* **[energy-advisor.py](energy-advisor.py)** - a command line agent entirely developed using ChatGPT Pro and Cursor which is able to autogenerate an HTML insights report plus graph from the input csv. Various models are supported from OpenAI (`gpt-4` and `gpt-3.5-turbo`), Groq (`mixtral-8x7b-32768` and `llama2-70b-4096`) and Ollama (`llama3.2`).  The command line looks like this:
```
Usage:
    energy-advisor.py [-v] [-m MODEL] [-i CONTEXT] <csv_file>
    energy-advisor.py (-h | --help)
    energy-advisor.py (-V | --version)

Options:
    -h --help           Show this help message
    -v --verbose        Enable verbose logging output
    -V --version        Show version and author information
    -m --model MODEL    Model to use for analysis [default: gpt-4]
                        Can be 'gpt-4', 'gpt-3.5-turbo', 'mixtral-8x7b-32768',
                        'llama2-70b-4096', or 'llama3.2' for local Ollama model
    -i --input CONTEXT  Path to text file containing user context for personalization

Arguments:
    csv_file            Path to CSV file containing energy data
```
Example usage invoking `groq` `mixtral-8x7b-32768` LLM using freeform input context supplied in the `context.txt` and bill data held in `combined-consumption-2024-01-01-2024-12-31.csv` file: 
```
$ python energy-advisor.py -m mixtral-8x7b-32768 combined-consumption-2024-01-01-2024-12-31.csv -i context.txt
```
Generated report from that command line looks like this:

<img width="485" alt="image" src="https://github.com/user-attachments/assets/90e8964d-5baa-4492-b571-490b5e2a192a" />

