import os
import sys
import math
import random
import statistics
import subprocess
import importlib
from pathlib import Path

# ------------------------------------------------------------
# Auto-install missing packages used by this script.
# ------------------------------------------------------------
def ensure_package(package_name: str) -> None:
    try:
        importlib.import_module(package_name)
    except ModuleNotFoundError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])

ensure_package("simpy")
ensure_package("reportlab")

import simpy
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

# ------------------------------------------------------------
# User-requested PDF output location and file name.
# ------------------------------------------------------------
OUTPUT_DIR = r"C:\Users\mstout\OneDrive - AANP\Documents\Workspace.MAIN\edX\GTx.ISYE6501\homework"
OUTPUT_FILE = "homework9_answers_13.1.pdf"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, OUTPUT_FILE)

# ------------------------------------------------------------
# Simulation settings from Homework 9, Question 13.2.
# ------------------------------------------------------------
ARRIVAL_RATE = 5.0           # passengers per minute
ID_MEAN_SERVICE = 0.75       # minutes
SCAN_MIN = 0.5               # minutes
SCAN_MAX = 1.0               # minutes
SIM_TIME = 1200              # minutes simulated per replication
WARMUP = 200                 # warm-up time ignored in metrics
REPLICATIONS = 10            # number of independent runs per staffing combo
MAX_SERVERS_TO_TEST = 6      # search from 1..6 at each stage
BASE_SEED = 6501
WAIT_TARGET = 15.0           # minutes


class AirportSecuritySystem:
    def __init__(self, env: simpy.Environment, num_id_checkers: int, num_scanners: int, rng: random.Random):
        self.env = env
        self.rng = rng
        self.id_check = simpy.Resource(env, capacity=num_id_checkers)
        self.scanners = [simpy.Resource(env, capacity=1) for _ in range(num_scanners)]
        self.total_wait_times = []

    def choose_shortest_scanner(self):
        queue_lengths = [len(scanner.queue) + scanner.count for scanner in self.scanners]
        shortest = min(queue_lengths)
        tied = [i for i, value in enumerate(queue_lengths) if value == shortest]
        chosen_index = self.rng.choice(tied)
        return self.scanners[chosen_index]

    def passenger(self, passenger_id: int):
        arrival_time = self.env.now

        with self.id_check.request() as req:
            id_queue_enter = self.env.now
            yield req
            wait_id = self.env.now - id_queue_enter
            service_id = self.rng.expovariate(1.0 / ID_MEAN_SERVICE)
            yield self.env.timeout(service_id)

        scanner = self.choose_shortest_scanner()
        with scanner.request() as req:
            scan_queue_enter = self.env.now
            yield req
            wait_scan = self.env.now - scan_queue_enter
            service_scan = self.rng.uniform(SCAN_MIN, SCAN_MAX)
            yield self.env.timeout(service_scan)

        total_wait = wait_id + wait_scan
        if arrival_time >= WARMUP:
            self.total_wait_times.append(total_wait)


def arrival_process(env: simpy.Environment, system: AirportSecuritySystem):
    passenger_id = 0
    while True:
        interarrival_time = system.rng.expovariate(ARRIVAL_RATE)
        yield env.timeout(interarrival_time)
        passenger_id += 1
        env.process(system.passenger(passenger_id))


def run_one_replication(num_id_checkers: int, num_scanners: int, seed: int) -> float:
    rng = random.Random(seed)
    env = simpy.Environment()
    system = AirportSecuritySystem(env, num_id_checkers, num_scanners, rng)
    env.process(arrival_process(env, system))
    env.run(until=SIM_TIME)

    if not system.total_wait_times:
        return float("inf")

    return statistics.mean(system.total_wait_times)


def evaluate_configurations():
    results = []

    for num_id_checkers in range(1, MAX_SERVERS_TO_TEST + 1):
        for num_scanners in range(1, MAX_SERVERS_TO_TEST + 1):
            replication_means = []

            for replication in range(REPLICATIONS):
                seed = BASE_SEED + 1000 * num_id_checkers + 100 * num_scanners + replication
                avg_wait = run_one_replication(num_id_checkers, num_scanners, seed)
                replication_means.append(avg_wait)

            mean_wait = statistics.mean(replication_means)
            std_wait = statistics.stdev(replication_means) if len(replication_means) > 1 else 0.0

            results.append({
                "id_checkers": num_id_checkers,
                "scanners": num_scanners,
                "mean_wait": mean_wait,
                "std_wait": std_wait,
            })

    feasible = [
        row for row in results
        if row["mean_wait"] < WAIT_TARGET
    ]

    feasible_sorted = sorted(
        feasible,
        key=lambda row: (
            row["id_checkers"] + row["scanners"],
            row["id_checkers"],
            row["scanners"],
            row["mean_wait"],
        )
    )

    best = feasible_sorted[0] if feasible_sorted else None
    return results, best


def build_pdf(results, best):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    doc = SimpleDocTemplate(
        OUTPUT_PATH,
        pagesize=letter,
        rightMargin=0.7 * inch,
        leftMargin=0.7 * inch,
        topMargin=0.7 * inch,
        bottomMargin=0.7 * inch,
    )

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="TitleCenter",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=17,
        leading=22,
        spaceAfter=12,
    ))
    styles.add(ParagraphStyle(
        name="BodyTight",
        parent=styles["BodyText"],
        fontSize=10.5,
        leading=14,
        spaceAfter=8,
    ))
    styles.add(ParagraphStyle(
        name="HeadingSmall",
        parent=styles["Heading2"],
        fontSize=12,
        leading=15,
        spaceBefore=8,
        spaceAfter=6,
    ))

    story = []
    story.append(Paragraph("Homework 9 - Questions 13.1 and 13.2", styles["TitleCenter"]))
    story.append(Paragraph(
        "This PDF was generated automatically by a Python script. The file name was set to the exact name requested: homework9_answers_13.1.pdf.",
        styles["BodyTight"],
    ))

    story.append(Paragraph("Question 13.1", styles["HeadingSmall"]))
    q131_text = (
        "<b>Binomial:</b> Number of customers out of 20 contacted who agree to schedule a product demo.<br/>"
        "<b>Geometric:</b> Number of sales calls made until the first successful sale.<br/>"
        "<b>Poisson:</b> Number of cars arriving at a drive-through in a 10-minute period.<br/>"
        "<b>Exponential:</b> Time between consecutive customer arrivals at a service desk.<br/>"
        "<b>Weibull:</b> Lifetime of a machine part before failure."
    )
    story.append(Paragraph(q131_text, styles["BodyTight"]))

    story.append(Paragraph("Question 13.2", styles["HeadingSmall"]))
    methodology = (
        "Passengers arrive according to a Poisson process with rate 5 per minute. They first wait in one common "
        "ID/boarding-pass queue with multiple servers. After that, each passenger joins the shortest personal-check "
        "queue, where screening time is uniformly distributed from 0.5 to 1.0 minutes. I tested staffing combinations "
        "from 1 to 6 ID checkers and 1 to 6 personal-check queues. For each combination, I ran multiple replications "
        "and computed the average total waiting time."
    )
    story.append(Paragraph(methodology, styles["BodyTight"]))

    if best is None:
        conclusion_text = (
            "No tested configuration kept the average total waiting time below 15 minutes. "
            "Increase MAX_SERVERS_TO_TEST in the script and rerun it."
        )
    else:
        conclusion_text = (
            f"The smallest staffing configuration that kept the average total waiting time below 15 minutes was "
            f"<b>{best['id_checkers']} ID/boarding-pass checkers</b> and "
            f"<b>{best['scanners']} personal-check queues</b>. "
            f"Its estimated average waiting time was <b>{best['mean_wait']:.2f} minutes</b>."
        )
    story.append(Paragraph(conclusion_text, styles["BodyTight"]))

    story.append(Spacer(1, 0.12 * inch))
    story.append(Paragraph("Simulation Results", styles["HeadingSmall"]))

    table_data = [["ID Checkers", "Personal Queues", "Mean Wait (min)", "Std. Dev."]]
    for row in sorted(results, key=lambda r: (r["id_checkers"], r["scanners"])):
        table_data.append([
            str(row["id_checkers"]),
            str(row["scanners"]),
            f"{row['mean_wait']:.2f}",
            f"{row['std_wait']:.2f}",
        ])

    table = Table(table_data, colWidths=[1.2 * inch, 1.35 * inch, 1.45 * inch, 1.15 * inch], repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f3a5f")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("LEADING", (0, 0), (-1, -1), 12),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.HexColor("#eef3f8")]),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#8ca0b3")),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(table)

    story.append(Spacer(1, 0.18 * inch))
    story.append(Paragraph(
        "Interpretation: three servers are not enough at either stage because average service capacity is below the arrival rate. "
        "A fourth server at both stages stabilizes the system and brings the average wait time well below the 15-minute requirement.",
        styles["BodyTight"],
    ))

    doc.build(story)


def main():
    results, best = evaluate_configurations()
    build_pdf(results, best)
    print(f"PDF created: {OUTPUT_PATH}")
    if best is not None:
        print(
            "Best configuration under 15 minutes: "
            f"{best['id_checkers']} ID checkers, {best['scanners']} personal queues, "
            f"mean wait = {best['mean_wait']:.2f} minutes"
        )


if __name__ == "__main__":
    main()
