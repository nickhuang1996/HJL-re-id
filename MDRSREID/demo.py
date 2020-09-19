from MDRSREID.Trainer.pre_initialization import pre_initialization
from MDRSREID.Trainer.MDRSReIDTrainer import MDRSReIDTrainer

if __name__ == '__main__':
    cfg = pre_initialization()
    trainer = MDRSReIDTrainer(cfg)
    if trainer.cfg.only_test is False:
        print("Train...")
        trainer.train()
    elif trainer.cfg.vis.use is True:
        print("visualize...")
        trainer.visualize()
    else:
        print("Test...")
        trainer.test()
