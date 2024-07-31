import main
import pickle


def replay_genome(genome_path="winner.pkl"):
    with open(genome_path, "rb") as f:
        winner_net = pickle.load(f)
    # Call game with only the loaded genome
    main.play_winner(winner_net, 60)


replay_genome()
