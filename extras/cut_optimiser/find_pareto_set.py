import argparse
import csv
import logging
import pathlib


log = logging.getLogger("find_pareto_set")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "db",
        type=pathlib.Path,
        help="the CSV database file",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        help="enable verbose output",
        action="store_true",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if (args.verbose or False) else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    results = []

    total_results = 0

    with open(args.db, "r") as f:
        reader = csv.DictReader(f)
        for i in reader:
            total_results += 1
            if i["success"] != "0":
                results.append({k: float(v) for k, v in i.items()})

    log.info(
        "Database contained %d results of which %d are valid",
        total_results,
        len(results),
    )

    pareto_set = []

    for i, m in enumerate(results):
        for j, n in enumerate(results):
            if i == j:
                continue

            if (
                n["rec_throughput"] <= m["rec_throughput"]
                and n["efficiency"] >= m["efficiency"]
                and n["fake_rate"] <= m["fake_rate"]
                and n["duplicate_rate"] <= m["duplicate_rate"]
                and n["seeding_efficiency"] >= m["seeding_efficiency"]
                and n["seed_fake_rate"] <= m["seed_fake_rate"]
                and n["seed_duplicate_rate"] <= m["seed_duplicate_rate"]

            ):
                log.debug(
                    "Removing %s from the Pareto set because %s is superior",
                    str(n),
                    str(m),
                )
                break
        else:
            pareto_set.append(m)

    log.info("Pareto set contains %d elements:", len(pareto_set))

    for i in sorted(pareto_set, key=lambda x: x["rec_throughput"], reverse=True):
        log.info(
            "  Eff. %.2f, fake rate %.2f, duplicate rate %.2f with reciprocal througput %.1fms and  Seeding Eff. %.2f, Seed fake rate %.2f, Seed duplicate rate %.2f is achieved by setup {%s}",
            100.0 * i["efficiency"],
            i["fake_rate"],
            i["duplicate_rate"],
            i["rec_throughput"] * 1000.0,
            i["seeding_efficiency"],
            i["seed_fake_rate"],
            i["seed_duplicate_rate"],
            ", ".join(
                "%s: %s" % (k, str(v))
                for k, v in i.items()
                if k
                not in [
                    "efficiency",
                    "fake_rate",
                    "duplicate_rate",
                    "rec_throughput",
                    "seeding_efficiency",
                    "seed_fake_rate",
                    "seed_duplicate_rate",
                    "success",
                ]
            ),
        )


if __name__ == "__main__":
    main()
