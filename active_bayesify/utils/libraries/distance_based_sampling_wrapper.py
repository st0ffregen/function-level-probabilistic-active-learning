import subprocess
import shlex


class DistanceBasedSamplingWrapper:

    def sample(self, container_name: str):
        """
        Samples x264 configurations with distance based sampling
        # TODO: make generic so that every software system can be executed
        # TODO: find out what to use the runs (like 42) are for and if we should use several of them. Update: this is the random seed
        # TODO: Tweak output of runRandomHundredTimes script to include more samples if needed
        # TODO: if time left, implement runRandomHundredTimes myself with numeric options (then turn "onlyBinary:True" off)

        :param container_name: name of the docker container.
        """
        start_container = shlex.split(f"docker start {container_name}")
        subprocess.run(start_container)

        run_sampling = shlex.split(f"docker exec -w /application/Distance-Based_Data {container_name} ./SPLConquerorExecuter.py x264 divDistBased /application/Distance-Based_Data/SupplementaryWebsite/PerformancePredictions/NewRuns 42 42")
        subprocess.run(run_sampling)

        read_data = shlex.split(f"docker exec -w /application/Distance-Based_Data {container_name} cat ./SupplementaryWebsite/PerformancePredictions/NewRuns/x264/x264_42/sampledConfigurations_divDistBased_t3.csv")
        results = subprocess.check_output(read_data)

        return results
