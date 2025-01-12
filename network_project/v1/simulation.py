# simulation.py
from collections import deque
import numpy as np
import matplotlib.pyplot as plt

from config import CONFIG

class Queue:
    def __init__(self, service_rate, servers,
                 time_distribution='exponential',
                 fixed_time=None, policy='FIFO'):
        self.service_rate = service_rate
        self.servers = servers
        self.queue = deque()
        self.in_service = []
        self.time_distribution = time_distribution
        self.fixed_time = fixed_time
        self.queue_history = []
        self.policy = policy

    def _get_service_time(self):
        if self.time_distribution == 'exponential':
            service_time = np.random.exponential(1 / self.service_rate)
        else:
            # fixed
            service_time = self.fixed_time + np.random.normal(0, 1 / self.service_rate)
        return max(service_time, 1/(10*self.service_rate))

    def add_customer(self, customer):
        if len(self.in_service) < self.servers:
            self.in_service.append(customer)
        else:
            if self.policy == 'FIFO':
                self.queue.append(customer)
            elif self.policy == 'LIFO':
                self.queue.appendleft(customer)
            elif self.policy == 'LIFO + PR':
                self.queue.append(customer)
                self.queue = deque(
                    sorted(self.queue, key=lambda x: x['class'], reverse=True)
                )

    def process(self, time_step):
        completed = []
        # obsługa w in_service
        for cust in self.in_service:
            cust["remaining_time"] -= time_step
        for cust in list(self.in_service):
            if cust["remaining_time"] <= 0:
                completed.append(cust)
                self.in_service.remove(cust)
        # przenoszenie z kolejki
        while len(self.in_service) < self.servers and self.queue:
            nxt = self.queue.popleft()
            nxt["remaining_time"] = self._get_service_time()
            self.in_service.append(nxt)

        self.queue_history.append(len(self.queue))
        return completed


class HospitalNetwork:
    def __init__(self, config):
        self.config = config
        # Tworzymy obiekty Queue:
        self.registration = Queue(
            service_rate=config['registration_rate'],
            servers=config['registration_servers'],
            time_distribution=config['registration_distribution'],
            fixed_time=config['registration_fixed_time'],
            policy='FIFO'
        )
        self.admissions = Queue(
            service_rate=config['admissions_rate'],
            servers=config['admissions_servers'],
            policy='LIFO'
        )
        self.gynecology = Queue(
            service_rate=config['gynecology_rate'],
            servers=config['gynecology_servers'],
            policy='FIFO'
        )
        self.delivery = Queue(
            service_rate=config['delivery_rate'],
            servers=config['delivery_servers'],
            policy='LIFO'
        )
        self.icu = Queue(
            service_rate=config['icu_rate'],
            servers=config['icu_servers'],
            policy='FIFO'
        )
        self.postpartum = Queue(
            service_rate=config['postpartum_rate'],
            servers=config['postpartum_servers'],
            policy='FIFO'
        )

        self.customers = []
        self.time = 0.0

    def add_patient(self, arrival_time, patient_class):
        self.customers.append({
            "arrival_time": arrival_time,
            "remaining_time": 0.0,
            "class": patient_class,  # 1,2,3
            "location": None
        })

    def simulate(self, total_time, time_step=0.1):
        cfg = self.config  # skrót

        while self.time < total_time:
            comp_reg = self.registration.process(time_step)
            comp_adm = self.admissions.process(time_step)
            comp_gyn = self.gynecology.process(time_step)
            comp_del = self.delivery.process(time_step)
            comp_icu = self.icu.process(time_step)
            comp_post = self.postpartum.process(time_step)

            # Rejestracja -> Izba przyjęć
            for cust in comp_reg:
                cust["remaining_time"] = max(
                    np.random.exponential(1 / self.admissions.service_rate),
                    1/(10*self.admissions.service_rate)
                )
                cust["location"] = "admissions"
                self.admissions.add_customer(cust)

            # Izba przyjęć
            for cust in comp_adm:
                ccls = cust["class"]  # 1,2,3
                if ccls == 1:
                    # p_out = 0.1
                    p_out = cfg['admissions_class1_out']  
                    if np.random.rand() < p_out:
                        cust["location"] = "discharged"
                    else:
                        # do delivery
                        cust["remaining_time"] = max(
                            np.random.exponential(1 / self.delivery.service_rate),
                            1/(10*self.delivery.service_rate)
                        )
                        cust["location"] = "delivery"
                        self.delivery.add_customer(cust)

                elif ccls == 3:
                    # w 100% do delivery (w config np. p_out=0)
                    p_out = cfg['admissions_class3_out']
                    if np.random.rand() < p_out:
                        cust["location"] = "discharged"
                    else:
                        cust["remaining_time"] = max(
                            np.random.exponential(1 / self.delivery.service_rate),
                            1/(10*self.delivery.service_rate)
                        )
                        cust["location"] = "delivery"
                        self.delivery.add_customer(cust)

                elif ccls == 2:
                    # p_out=0.2, p_del=0.4, p_gyn=0.4
                    p_out = cfg['admissions_class2_out']
                    p_del = cfg['admissions_class2_delivery']
                    # p_gyn = 1 - p_out - p_del => config['admissions_class2_gyn']
                    p_gyn = cfg['admissions_class2_gyn']
                    r = np.random.rand()
                    if r < p_out:
                        cust["location"] = "other_department"
                    elif r < p_out + p_del:
                        cust["remaining_time"] = max(
                            np.random.exponential(1 / self.delivery.service_rate),
                            1/(10*self.delivery.service_rate)
                        )
                        cust["location"] = "delivery"
                        self.delivery.add_customer(cust)
                    else:
                        cust["remaining_time"] = max(
                            np.random.exponential(1 / self.gynecology.service_rate),
                            1/(10*self.gynecology.service_rate)
                        )
                        cust["location"] = "gynecology"
                        self.gynecology.add_customer(cust)

            # Ginekologia
            for cust in comp_gyn:
                if cust["class"] == 2:
                    p_out = cfg['gyn_class2_out']  # 0.1
                    if np.random.rand() < p_out:
                        cust["location"] = "discharged"
                    else:
                        # do delivery
                        cust["remaining_time"] = max(
                            np.random.exponential(1 / self.delivery.service_rate),
                            1/(10*self.delivery.service_rate)
                        )
                        cust["location"] = "delivery"
                        self.delivery.add_customer(cust)

            # Porodówka + zmiana klasy
            for cust in comp_del:
                old_class = cust["class"]
                rr = np.random.rand()
                if old_class == 3:  # 3->2(0.4),1(0.3),3(0.3)
                    p_3to2 = cfg['delivery_class3_to2']
                    p_3to1 = cfg['delivery_class3_to1']
                    if rr < p_3to2:
                        new_class = 2
                    elif rr < p_3to2 + p_3to1:
                        new_class = 1
                    else:
                        new_class = 3
                elif old_class == 2:  # 2->1(0.6),3(0.1),2(0.3)
                    p_2to1 = cfg['delivery_class2_to1']
                    p_2to3 = cfg['delivery_class2_to3']
                    if rr < p_2to1:
                        new_class = 1
                    elif rr < p_2to1 + p_2to3:
                        new_class = 3
                    else:
                        new_class = 2
                elif old_class == 1:  # 1->3(0.03),2(0.07),1(0.9)
                    p_1to3 = cfg['delivery_class1_to3']
                    p_1to2 = cfg['delivery_class1_to2']
                    if rr < p_1to3:
                        new_class = 3
                    elif rr < p_1to3 + p_1to2:
                        new_class = 2
                    else:
                        new_class = 1

                cust["class"] = new_class
                # kl3 => icu, kl1/2 => postpartum
                if new_class == 3:
                    cust["remaining_time"] = max(
                        np.random.exponential(1 / self.icu.service_rate),
                        1/(10*self.icu.service_rate)
                    )
                    cust["location"] = "icu"
                    self.icu.add_customer(cust)
                else:
                    cust["remaining_time"] = max(
                        np.random.exponential(1 / self.postpartum.service_rate),
                        1/(10*self.postpartum.service_rate)
                    )
                    cust["location"] = "postpartum"
                    self.postpartum.add_customer(cust)

            # ICU
            for cust in comp_icu:
                p_newclass1 = cfg['icu_p_newclass1']  # 0.5
                # choose [1,2] z tymi p
                if np.random.rand() < p_newclass1:
                    new_class = 1
                    cust["remaining_time"] = max(
                        np.random.exponential(1 / self.postpartum.service_rate),
                        1/(10*self.postpartum.service_rate)
                    )
                    cust["location"] = "postpartum"
                    self.postpartum.add_customer(cust)
                else:
                    new_class = 2
                    cust["location"] = "other_department"
                cust["class"] = new_class

            # postpartum
            for cust in comp_post:
                if cust["class"] == 1:
                    cust["location"] = "discharged"
                elif cust["class"] == 2:
                    # p=0.7 => kl1 => out
                    p_2to1 = cfg['postpartum_class2_switch_to1']
                    if np.random.rand() < p_2to1:
                        cust["class"] = 1
                        cust["location"] = "discharged"
                    else:
                        cust["location"] = "other_department"

            # Nowi pacjenci w tym kroku
            new_custs = [
                c for c in self.customers
                if abs(c["arrival_time"] - self.time) < 1e-9
            ]
            for nc in new_custs:
                if nc["class"] == 3:
                    # omija rejestrację
                    nc["remaining_time"] = max(
                        np.random.exponential(1 / self.admissions.service_rate),
                        1/(10*self.admissions.service_rate)
                    )
                    nc["location"] = "admissions"
                    self.admissions.add_customer(nc)
                else:
                    # kl1/kl2 => rejestracja
                    nc["location"] = "registration"
                    self.registration.add_customer(nc)

            self.time += time_step

        print("Symulacja zakończona.")

    def plot_queue_lengths(self):
        plt.figure(figsize=(10,5))
        plt.plot(self.registration.queue_history, label='Rejestracja')
        plt.plot(self.admissions.queue_history, label='Izba przyjęć')
        plt.plot(self.gynecology.queue_history, label='Ginekologia')
        plt.plot(self.delivery.queue_history, label='Porodówka')
        plt.plot(self.icu.queue_history, label='OIOM')
        plt.plot(self.postpartum.queue_history, label='Poporodowa')
        plt.title('Długości kolejek w czasie')
        plt.xlabel('Krok symulacji')
        plt.ylabel('Liczba oczekujących')
        plt.legend()
        plt.grid(True)
        plt.show()
