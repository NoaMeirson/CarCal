
#buildAnalyzeRequest(file)
##gets image file from user, converts it to base64 format, builds an object from AnalyzeRequest scheme
##input: file, output: JSON file


#sendAnalyzeRequest()
##gets the JSON object, doing JSON.stringify, and sends the object to the correct endpoint in the API
##the enpoint in the API is /analyze but please define it in the config file and in the code justtake it from the config


