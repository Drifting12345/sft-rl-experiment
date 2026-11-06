"""Thin Ray driver using verl's synchronous trainer and agent manager."""
import ray


@ray.remote(num_cpus=1)
class RLTaskRunner:
    def run(self, config):
        import transfer_queue as tq
        from verl.trainer.ppo.v1 import AgentLoopManagerTQ
        from verl.utils.logging_utils import configure_verl_logging
        from trainer import IGSMTrainer

        configure_verl_logging()
        config.transfer_queue.enable = True
        tq.init(config.transfer_queue)
        trainer = None
        succeeded = False
        try:
            if config.igsm_rl.get("epoch_unit") == "data-pass":
                from single_group_trainer import SingleGroupEpochTrainer
                trainer = SingleGroupEpochTrainer(config)
            else:
                trainer = IGSMTrainer(config)
            trainer.init()
            manager = AgentLoopManagerTQ.create(config=config, llm_client=trainer.get_llm_client(),
                teacher_client=trainer.get_teacher_client(), reward_loop_worker_handles=trainer.get_reward_handles())
            trainer.fit(manager)
            succeeded = True
        finally:
            try:
                logger = getattr(trainer, "logger", None)
                if logger is not None:
                    logger.finish(exit_code=0 if succeeded else 1)
            finally:
                tq.close()
