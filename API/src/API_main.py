#analyze(request)
##def analyze(request: AnalyzeRequest): i explained it in the page i have sent in the group. 
##this is the function that gets the model built with AnalyzeRequest scheme. this function operates the
##validate_image(), resize_image(), send_to_engine()

#validate_image()
##makes sure that the image is:base64, right format, right size. not using any scheme since we already have the model.

#resize_image()
##if the image is not in the correct size, we will resize it. for exapmle the YOLOmodel needs 640x640 size

#send_to_engine()
##def send_to_engine(engine_request: EngineAnalyzeRequest):
##building an JSON object from the model -using the scheme EngineAnalyzeRequest, 
## sends HTTP POST to the engine and needs to get EngineAnalyzeResponse from the engine

#health()