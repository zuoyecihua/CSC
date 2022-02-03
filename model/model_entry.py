from model.corrector.bart import BartCorrector
from model.judger.base import SentenceJudger
from model.corrector.base import ErrCorrector
from model.promptor.base import ErrPromptJudger
from model.corrector.pinyin import ErrPinyinCorrector
from model.corrector.pinyin_semantic import PinyinSemanticCorrector
from model.detec_correct_pipeline.binary import DetecSelectPipeline

def select_model(args):
    type2model = {
        'SentenceJudger': SentenceJudger,
        "ErrCorrector": ErrCorrector,
        'ErrPromptJudger': ErrPromptJudger,
        'ErrPinyinCorrector': ErrPinyinCorrector,
        'PinyinSemanticCorrector': PinyinSemanticCorrector,
        'DetecSelectPipeline': DetecSelectPipeline,
        'BartCorrector': BartCorrector,
    }
    model = type2model[args.model_type](args)
    return model

