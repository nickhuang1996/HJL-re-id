from collections import OrderedDict
from MDRSREID.Analyze_Meter.accuracy_computer import AccuracyComputer
from MDRSREID.Analyze_Meter.verification_probability_analyze import VerificationProbabilityAnalyze


def analyze_function_creation(cfg, tb_writer):
    analyze_functions = OrderedDict()

    if cfg.analyze_computer.use:
        analyze_functions[cfg.analyze_computer.name] = AccuracyComputer(cfg.analyze_computer, tb_writer)
    if cfg.verification_probability_analyze.use:
        analyze_functions[cfg.verification_probability_analyze.name] = VerificationProbabilityAnalyze(cfg.verification_probability_analyze, tb_writer)
    return analyze_functions
