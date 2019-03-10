

class InteractionSimulator:
    def __init__(
            self,
            interaction,
            self_interaction=None,
            input_dir=".",
            output_dir="."
            ):
        # Sanitize input_dir variable.
        input_dir = os.path.abspath(input_dir)
        logger.info("Particle advection input directory: {:s}".format(input_dir))

        # Sanitize output_dir variable.
        output_dir = os.path.abspath(output_dir)
        logger.info("Microbe interactions output directory: {:s}".format(output_dir))

        # Create output directory if it doesn't exist.
        if not os.path.exists(output_dir):
            logger.info("Creating directory: {:s}".format(output_dir))
            os.makedirs(output_dir)
