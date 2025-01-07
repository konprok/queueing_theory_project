import simpy
import random
import matplotlib.pyplot as plt

# Define constants
NORMAL_BIRTH = 'normal'
COMPLICATION = 'complication'
CRITICAL = 'critical'

class HospitalSimulation:
    def __init__(self, env):
        self.env = env

        # Define hospital resources
        self.registration = simpy.Resource(env, capacity=1)  # Registration desk
        self.admission = simpy.PriorityResource(env, capacity=2)  # Admission room
        self.delivery_room = simpy.PriorityResource(env, capacity=1)  # Delivery room
        self.postpartum = simpy.Resource(env, capacity=3)  # Postpartum room
        self.oiom = simpy.PriorityResource(env, capacity=1)  # ICU
        self.gynecology = simpy.Resource(env, capacity=2)  # Gynecology

        # For tracking queue lengths
        self.queue_lengths = {
            'registration': [],
            'admission': [],
            'delivery_room': [],
            'postpartum': [],
            'oiom': [],
            'gynecology': []
        }

    def update_queue_lengths(self):
        self.queue_lengths['registration'].append(len(self.registration.queue))
        self.queue_lengths['admission'].append(len(self.admission.queue))
        self.queue_lengths['delivery_room'].append(len(self.delivery_room.queue))
        self.queue_lengths['postpartum'].append(len(self.postpartum.queue))
        self.queue_lengths['oiom'].append(len(self.oiom.queue))
        self.queue_lengths['gynecology'].append(len(self.gynecology.queue))

    def process_registration(self, patient):
        yield self.env.timeout(random.expovariate(1 / 5))  # Mean of 5 minutes

    def process_admission(self, patient):
        yield self.env.timeout(random.uniform(8, 12))  # Uniform distribution, 8-12 minutes

    def process_delivery(self, patient):
        yield self.env.timeout(random.uniform(15, 25))  # Uniform distribution, 15-25 minutes

    def process_postpartum(self, patient):
        yield self.env.timeout(random.expovariate(1 / 15))  # Mean of 15 minutes

    def process_oiom(self, patient):
        yield self.env.timeout(random.uniform(25, 35))  # Uniform distribution, 25-35 minutes

    def process_gynecology(self, patient):
        yield self.env.timeout(random.uniform(10, 20))  # Uniform distribution, 10-20 minutes


# Define a patient
class Patient:
    def __init__(self, env, hospital, patient_type):
        self.env = env
        self.hospital = hospital
        self.type = patient_type

    def run(self):
        # Registration process
        with self.hospital.registration.request() as req:
            yield req
            yield self.env.process(self.hospital.process_registration(self))
        self.hospital.update_queue_lengths()

        # Admission process
        with self.hospital.admission.request(priority=self.priority()) as req:
            yield req
            yield self.env.process(self.hospital.process_admission(self))
        self.hospital.update_queue_lengths()

        # Route patient
        if self.type == NORMAL_BIRTH:
            with self.hospital.delivery_room.request(priority=self.priority()) as req:
                yield req
                yield self.env.process(self.hospital.process_delivery(self))
            self.hospital.update_queue_lengths()

            with self.hospital.postpartum.request() as req:
                yield req
                yield self.env.process(self.hospital.process_postpartum(self))
            self.hospital.update_queue_lengths()

        elif self.type == COMPLICATION:
            with self.hospital.gynecology.request() as req:
                yield req
                yield self.env.process(self.hospital.process_gynecology(self))
            self.hospital.update_queue_lengths()

        elif self.type == CRITICAL:
            with self.hospital.oiom.request(priority=self.priority()) as req:
                yield req
                yield self.env.process(self.hospital.process_oiom(self))
            self.hospital.update_queue_lengths()

    def priority(self):
        if self.type == CRITICAL:
            return 1
        elif self.type == COMPLICATION:
            return 2
        else:
            return 3


# Simulation setup
def run_simulation(duration, normal_rate, complication_rate, critical_rate):
    env = simpy.Environment()
    hospital = HospitalSimulation(env)

    def generate_patients():
        while True:
            patient_type = random.choices(
                [NORMAL_BIRTH, COMPLICATION, CRITICAL],
                weights=[normal_rate, complication_rate, critical_rate],
            )[0]
            patient = Patient(env, hospital, patient_type)
            env.process(patient.run())
            yield env.timeout(random.expovariate(1 / 3))  # Mean arrival every 3 minutes

    env.process(generate_patients())
    env.run(until=duration)

    # Plot queue lengths
    plt.figure(figsize=(10, 6))
    for key, values in hospital.queue_lengths.items():
        plt.plot(values, label=key)
    plt.xlabel('Time (minutes)')
    plt.ylabel('Queue Length')
    plt.title('Queue Lengths in Hospital Departments')
    plt.legend()
    plt.grid()
    plt.show()

# Example usage
run_simulation(240, normal_rate=5, complication_rate=2, critical_rate=1)
