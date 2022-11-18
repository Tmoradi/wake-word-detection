# Here we are creating a handler file for 
# preprocessing , inference , and postprocessing of our model.
import torch 
import logging
from models.cnn import MelSpectrogramNetwork
from ts.torch_handler.base_handler import BaseHandler
from torchvision.transforms import Compose , PILToTensor

logger = logging.getLogger(__name__)

class WakeWordHandler(BaseHandler):
    def __init__(self):
        self.model = None 
        self.device = None
        self.metrics = None   
    
    def initialize(self,ctx):
        # Here we are going to be loading in our pytorch lightning model
        properties = ctx.system_properties
        model_dir = properties.get("model_dir")
        if not torch.cuda.is_available() or properties.get("gpu_id") is None:
            self.device ='cpu' 
        self.device = torch.device("cuda:" + str(properties.get("gpu_id")))

        self.model = torch.jit.load('model-store/model.pt')
        logger.debug("Our Mel Spectrogram Network loaded successfully")
    
    def preprocess(self, data) -> torch.Tensor:
        # Here we are going to be loading in our dataset, 
        # we are going to do this, depending on the file we are getting 
        transforms = transforms.Compose([transforms.PILToTensor()])
        tensor = transforms(data)
        tensor = tensor.unsqueeze(0)
        return tensor 

    def inference(self, data):
        with torch.no_grad:
            data = data.to(device=self.device,type=torch.FloatTensor)
            pred = self.model(data)
            return pred 

    def postprocess(self, inference_output):
        output = inference_output.argmax(axis=1)
        confidence = inference_output.softmax(axis=1).flatten()[output].item()
        return [{'predicted_class':1,
                'confidence':confidence}]


