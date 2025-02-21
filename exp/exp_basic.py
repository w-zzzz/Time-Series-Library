import os
import torch
from models import Autoformer, Transformer, TimesNet, Nonstationary_Transformer, DLinear, FEDformer, \
    Informer, LightTS, Reformer, ETSformer, Pyraformer, PatchTST, MICN, Crossformer, FiLM, iTransformer, \
    Koopa, TiDE, FreTS, TimeMixer, TSMixer, SegRNN, MambaSimple, TemporalFusionTransformer, SCINet, PAttn, TimeXer, \
    WPMixer, MultiPatchFormer


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'TimesNet': TimesNet,
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Nonstationary_Transformer': Nonstationary_Transformer,
            'DLinear': DLinear,
            'FEDformer': FEDformer,
            'Informer': Informer,
            'LightTS': LightTS,
            'Reformer': Reformer,
            'ETSformer': ETSformer,
            'PatchTST': PatchTST,
            'Pyraformer': Pyraformer,
            'MICN': MICN,
            'Crossformer': Crossformer,
            'FiLM': FiLM,
            'iTransformer': iTransformer,
            'Koopa': Koopa,
            'TiDE': TiDE,
            'FreTS': FreTS,
            'MambaSimple': MambaSimple,
            'TimeMixer': TimeMixer,
            'TSMixer': TSMixer,
            'SegRNN': SegRNN,
            'TemporalFusionTransformer': TemporalFusionTransformer,
            "SCINet": SCINet,
            'PAttn': PAttn,
            'TimeXer': TimeXer,
            'WPMixer': WPMixer,
            'MultiPatchFormer': MultiPatchFormer
        }
        if args.model == 'Mamba':
            print('Please make sure you have successfully installed mamba_ssm')
            from models import Mamba
            self.model_dict['Mamba'] = Mamba

        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu and self.args.gpu_type == 'cuda':
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        elif self.args.use_gpu and self.args.gpu_type == 'mps':
            device = torch.device('mps')
            print('Use GPU: mps')
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device
    
    # def _acquire_device(self):
    #     """Acquire appropriate computing device based on arguments and availability."""
    #     if self.args.use_gpu:
    #         if self.args.gpu_type == 'cuda':
    #             # Check CUDA availability
    #             if not torch.cuda.is_available():
    #                 print('Warning: CUDA device requested but not available. Using CPU instead.')
    #                 return torch.device('cpu')
                
    #             if torch.cuda.is_available():
    #                 # Print detailed CUDA info
    #                 print(f"CUDA available: {torch.cuda.is_available()}")
    #                 print(f"Current CUDA device: {torch.cuda.current_device()}")
    #                 print(f"Device name: {torch.cuda.get_device_name(0)}")
    #                 print(f"Current device count: {torch.cuda.device_count()}")
                    
    #             # Set visible devices
    #             os.environ["CUDA_VISIBLE_DEVICES"] = str(
    #                 self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
                
    #             # Verify requested GPU exists
    #             if self.args.gpu >= torch.cuda.device_count():
    #                 print(f'Warning: GPU {self.args.gpu} not found. Using GPU 0.')
    #                 self.args.gpu = 0
                    
    #             device = torch.device(f'cuda:{self.args.gpu}')
    #             print(f'Use GPU: cuda:{self.args.gpu}')
                
    #         elif self.args.gpu_type == 'mps':
    #             # Check MPS availability (Apple Silicon)
    #             if not torch.backends.mps.is_available():
    #                 print('Warning: MPS device requested but not available. Using CPU instead.')
    #                 return torch.device('cpu')
                    
    #             device = torch.device('mps')
    #             print('Use GPU: mps')
                
    #         else:
    #             print(f'Warning: Unknown GPU type {self.args.gpu_type}. Using CPU instead.')
    #             device = torch.device('cpu')
    #     else:
    #         device = torch.device('cpu')
    #         print('Use CPU')
            
    #     return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
