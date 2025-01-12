# analysis.py

import numpy as np
import matplotlib.pyplot as plt
from math import factorial, exp
from config import CONFIG

def build_routing_matrix_4d(cfg):
    """
    Buduje 4D macierz p_{(i,old_class)->(j,new_class)} w sieci 6-węzłowej:
      i,j in {0..5}:
        0 = registration
        1 = admissions
        2 = gynecology
        3 = delivery
        4 = icu
        5 = postpartum

    Klasy w indexach:
       klasa1 => r=0
       klasa2 => r=1
       klasa3 => r=2

    Wypełniamy zgodnie z prawdopodobieństwami w config.py
    i logiką z 'simulate(...)'.
    """

    N=6
    R=3
    # p_{(i,r)->(j,s)}
    P = np.zeros((N, R, N, R))

    ########################################
    # 1) Rejestracja (node=0) -> Admissions(1)
    #
    #  W symulacji: kl1, kl2 => rejestracja -> admissions,
    #               kl3 omija rejestrację, ale w BCMP i tak
    #               e_{0,2} pewnie wyjdzie 0.
    #  Zrobimy: P[0,r_, 1, r_] = 1.0  (bez zmiany klasy)
    ########################################
    for r_ in range(R):
        P[0, r_, 1, r_] = 1.0

    ########################################
    # 2) Admissions (node=1)
    #
    #  - klasa1(r=0):
    #       p_out = cfg['admissions_class1_out']
    #       => (1 - p_out) do delivery (node=3), klasa ta sama (r=0)
    #
    #  - klasa2(r=1):
    #       p_out = cfg['admissions_class2_out']
    #       p_del = cfg['admissions_class2_delivery']
    #       p_gyn = cfg['admissions_class2_gyn']
    #
    #  - klasa3(r=2):
    #       p_out = cfg['admissions_class3_out'] (zwykle 0)
    #       => (1 - p_out) do delivery(3, r=2)
    ########################################

    out1 = cfg['admissions_class1_out']  # np.0.1
    # klasa1(r=0), do node=3, r=0 => (1-out1)
    P[1, 0, 3, 0] = 1.0 - out1

    # klasa2(r=1)
    out2 = cfg['admissions_class2_out']       # np.0.2
    p_del2 = cfg['admissions_class2_delivery']# np.0.4
    p_gyn2 = cfg['admissions_class2_gyn']     # np.0.4
    P[1, 1, 3, 1] = p_del2  # do delivery
    P[1, 1, 2, 1] = p_gyn2  # do gynecology

    # klasa3(r=2)
    out3 = cfg['admissions_class3_out']  # zwykle 0
    P[1, 2, 3, 2] = 1.0 - out3

    ########################################
    # 3) Ginekologia (2)
    #
    #  W simulate: klasa2 => p=0.1 out, 0.9 do delivery
    #  klasa2 => r=1
    #  P[2, 1, 3, 1] = 1.0 - 0.1 = 0.9
    ########################################
    out_g2 = cfg['gyn_class2_out']
    P[2, 1, 3, 1] = 1.0 - out_g2

    ########################################
    # 4) Delivery (3) + zmiana klasy
    #
    #   - old=kl3(r=2): 3->2(0.4),1(0.3),3(0.3) w simulate
    #     => w indexach: old=2 => new=1 p=0.4, new=0 p=0.3, new=2 p=0.3
    #     jeśli new=2 => ICU(4), w innym wypadku postpartum(5).
    ########################################
    p_3to2 = cfg['delivery_class3_to2']  # np.0.4
    p_3to1 = cfg['delivery_class3_to1']  # np.0.3
    # reszta => 0.3
    P[3, 2, 4, 2] = 1.0 - (p_3to2 + p_3to1)  # ICU
    P[3, 2, 5, 1] = p_3to2  # postpartum, new=1
    P[3, 2, 5, 0] = p_3to1  # postpartum, new=0

    #
    #   - old=kl2(r=1): 2->1(0.6),3(0.1),2(0.3)
    #
    p_2to1 = cfg['delivery_class2_to1']  # 0.6
    p_2to3 = cfg['delivery_class2_to3']  # 0.1
    P[3, 1, 4, 2] = p_2to3
    # postpartum
    P[3, 1, 5, 0] = p_2to1
    P[3, 1, 5, 1] = 1.0 - (p_2to1 + p_2to3)

    #
    #   - old=kl1(r=0): 1->3(0.03),2(0.07),1(0.9)
    #
    p_1to3 = cfg['delivery_class1_to3']  # 0.03
    p_1to2 = cfg['delivery_class1_to2']  # 0.07
    P[3, 0, 4, 2] = p_1to3
    P[3, 0, 5, 1] = p_1to2
    P[3, 0, 5, 0] = 1.0 - (p_1to3 + p_1to2)

    ########################################
    # 5) ICU(4)
    #   p=0.5 => new_class=1 => postpartum(5,0)
    #   p=0.5 => new_class=2 => out
    ########################################
    p_icu1 = cfg['icu_p_newclass1']
    for oldc in range(R):
        P[4, oldc, 5, 0] = p_icu1

    ########################################
    # 6) postpartum(5) => out
    ########################################

    return P


def _mmc_stationary_distribution(c, rho, max_k=20):
    """
    Zwraca rozkład stacjonarny pi(k) dla M/M/c,
    z parametrem rho = λ/μ (zakładamy rho < c).
    """
    if rho >= c:
        return None
    Z = 0.0
    for k in range(c):
        Z += (rho**k)/factorial(k)
    Z += (rho**c/factorial(c))*( c/(c-rho) )
    p0 = 1.0/Z
    dist=[]
    for k in range(max_k+1):
        if k < c:
            val = p0*(rho**k)/factorial(k)
        else:
            val = p0*(rho**c/factorial(c))*((1.0/c)**(k-c))
        dist.append(val)
    return dist


def compute_bcmp_open_network_4d(
    N, R,
    arrival_rates,      # [λ0, λ1, λ2]
    service_rates,      # shape=(N,R)
    node_types,         # np.array(N,) in {1,2,3,4}
    servers_per_node,   # np.array(N,)
    routing_matrix_4d,  # shape=(N,R,N,R)
    max_k_for_print=10
):
    """
    Obliczenia BCMP w sieci z przełączaniem klas (class switching).
    Zwraca słownik:
      {
        'e_i_r':   (N,R),
        'rho_i_r': (N,R),
        'rho_i':   (N,),
        'K_i_r':   (N,R),
        'K_i':     (N,),
        'pi_i_k':  dict: i -> np.array(k=0..max_k)
      }
    """
    # 1) p0[i,r] -> skąd klasa r wchodzi z zewnątrz?
    p0 = np.zeros((N,R))
    # klasa0=> node0, klasa1=> node0, klasa2=> node1
    p0[0,0] = 1.0
    p0[0,1] = 1.0
    p0[1,2] = 1.0

    # 2) e_{i,r} iteracyjnie
    e_i_r = np.copy(p0)
    for _iter in range(1000):
        new_e = np.copy(e_i_r)
        for i in range(N):
            for r_ in range(R):
                external_inflow = arrival_rates[r_]*p0[i,r_]
                internal_inflow = 0.0
                for j in range(N):
                    for s in range(R):
                        internal_inflow += e_i_r[j,s]*routing_matrix_4d[j,s,i,r_]
                new_e[i,r_] = external_inflow + internal_inflow
        if np.allclose(new_e, e_i_r, atol=1e-12):
            break
        e_i_r = new_e

    # 3) rho_{i,r} = (λ_r * e_{i,r}) / μ_{i,r}
    rho_i_r = np.zeros((N,R))
    for i in range(N):
        for r_ in range(R):
            mu_ = service_rates[i,r_]
            if mu_>0:
                rho_i_r[i,r_] = (arrival_rates[r_]* e_i_r[i,r_]) / mu_
    rho_i = np.sum(rho_i_r, axis=1)

    # 4) K_i_r
    K_i_r = np.zeros((N,R))
    for i in range(N):
        typ = node_types[i]
        if typ==1 or typ==4:  # M/M/1 lub PS
            if rho_i[i]<1.0:
                for r_ in range(R):
                    if rho_i[i]>0:
                        K_i_r[i,r_] = (rho_i_r[i,r_]/ rho_i[i]) * ( rho_i[i]/(1.0-rho_i[i]) )
                    else:
                        K_i_r[i,r_] = 0.0
            else:
                K_i_r[i,:] = np.inf
        elif typ==3:  # M/M/∞
            for r_ in range(R):
                K_i_r[i,r_] = rho_i_r[i,r_]
        elif typ==2:  # M/M/c
            c_ = servers_per_node[i]
            lambd_i= 0.0
            mu_list=[]
            for r_ in range(R):
                lambd_i += arrival_rates[r_]* e_i_r[i,r_]
                if service_rates[i,r_]>0:
                    mu_list.append(service_rates[i,r_])
            if len(mu_list)==0:
                K_i_r[i,:]=0
                continue
            mu_avg= np.mean(mu_list)
            rho_prime= lambd_i/(c_*mu_avg)
            if rho_prime>=1.0:
                K_i_r[i,:]= np.inf
            else:
                def erlang_c(c,a):
                    from math import factorial
                    num= (a**c)/ factorial(c)* (c/(c-a))
                    s_=0.0
                    for k_ in range(c):
                        s_+= (a**k_)/ factorial(k_)
                    s_+= num
                    p0=1.0/s_
                    return num*p0
                a= lambd_i/mu_avg
                C= erlang_c(c_,a)
                Lq= C*(rho_prime/(1-rho_prime))
                Ls= c_*rho_prime
                L= Lq + Ls
                sum_rhoi= np.sum(rho_i_r[i,:])
                if sum_rhoi>0:
                    for r_ in range(R):
                        alpha= rho_i_r[i,r_]/ sum_rhoi
                        K_i_r[i,r_]= alpha*L

    K_i= np.sum(K_i_r, axis=1)

    # 5) pi_i_k
    pi_i_k={}
    for i in range(N):
        typ = node_types[i]
        if typ in [1,4]:  # M/M/1 or PS
            r_ = rho_i[i]
            if r_<1.0:
                dist=[]
                for k_ in range(max_k_for_print+1):
                    dist.append( (1.0 - r_)*(r_**k_) )
                pi_i_k[i]= np.array(dist)
            else:
                pi_i_k[i]= None
        elif typ==3: # M/M/∞
            r_ = rho_i[i]
            dist=[]
            from math import factorial
            for k_ in range(max_k_for_print+1):
                val= exp(-r_)*(r_**k_)/ factorial(k_)
                dist.append(val)
            pi_i_k[i]= np.array(dist)
        elif typ==2: # M/M/c
            c_= servers_per_node[i]
            lambd_i=0.0
            mu_list=[]
            for r_ in range(R):
                lambd_i+= arrival_rates[r_]* e_i_r[i,r_]
                if service_rates[i,r_]>0:
                    mu_list.append(service_rates[i,r_])
            if len(mu_list)==0:
                pi_i_k[i]= None
                continue
            mu_avg= np.mean(mu_list)
            big_rho= lambd_i/ mu_avg
            if big_rho>= c_:
                pi_i_k[i]= None
            else:
                dist= _mmc_stationary_distribution(c_, big_rho, max_k_for_print)
                pi_i_k[i]= np.array(dist) if dist is not None else None
        else:
            pi_i_k[i]= None

    return {
        'e_i_r':   e_i_r,
        'rho_i_r': rho_i_r,
        'rho_i':   rho_i,
        'K_i_r':   K_i_r,
        'K_i':     K_i,
        'pi_i_k':  pi_i_k
    }


def run_bcmp_analysis():
    """
    Główna funkcja do uruchomienia BCMP z configu
    i wyświetlenia wyników.
    """
    cfg = CONFIG
    # 1) Budujemy macierz 4D
    routing_4d = build_routing_matrix_4d(cfg)
    N=6
    R=3

    # 2) arrival_rates
    lam = cfg['arrival_lambda']
    pcls= cfg['class_probs']
    arrival_rates= np.array([lam*pcls[0], lam*pcls[1], lam*pcls[2]])

    # 3) node_types => 0=>M/M/1, reszta=>M/M/c
    node_types = np.array([1,2,2,2,2,2], dtype=int)

    # 4) servers
    servers = np.array([
        cfg['registration_servers'],
        cfg['admissions_servers'],
        cfg['gynecology_servers'],
        cfg['delivery_servers'],
        cfg['icu_servers'],
        cfg['postpartum_servers']
    ])

    # 5) service_rates shape=(N,R)
    service_rates = np.zeros((N,R))
    service_rates[0,:] = cfg['registration_rate']
    service_rates[1,:] = cfg['admissions_rate']
    service_rates[2,:] = cfg['gynecology_rate']
    service_rates[3,:] = cfg['delivery_rate']
    service_rates[4,:] = cfg['icu_rate']
    service_rates[5,:] = cfg['postpartum_rate']

    # 6) Obliczamy BCMP
    result = compute_bcmp_open_network_4d(
        N=N,
        R=R,
        arrival_rates=arrival_rates,
        service_rates=service_rates,
        node_types=node_types,
        servers_per_node=servers,
        routing_matrix_4d=routing_4d,
        max_k_for_print=12
    )

    # Nazwy węzłów:
    node_labels = [
        "Rejestracja",
        "Izba przyjęć",
        "Ginekologia",
        "Porodówka",
        "OIOM",
        "Sala poporodowa"
    ]

    # --- (A) Wykres sumarycznego obciążenia węzłów (rho_i) ---
    rho = result['rho_i']
    plt.figure(figsize=(8, 5))
    plt.bar(range(N), rho, color='green')
    plt.xlabel("Węzeł")
    plt.ylabel("Obciążenie ρ_i")
    plt.title("Obciążenie węzłów BCMP (z class switching)")
    plt.xticks(range(N), node_labels, rotation=0)  # etykiety zamiast 0..5
    plt.grid(True)
    plt.show()

    # --------------- DODATKOWE KODY WYKRESÓW ---------------

    # Kolory i nazwy klas (zamiast "Klasa 0" itp.)
    class_names = ["Bez komplikacji", "Z komplikacjami", "Krytyczne"]
    class_colors = ["green", "orange", "red"]

    # e_{i,r}
    e_i_r = 1/cfg['arrival_lambda'] * result['e_i_r']  # shape = (N, R)
    fig, ax = plt.subplots(figsize=(8,5))

    index = np.arange(N)       # x-pozycje węzłów
    bar_width = 0.25           # szerokość słupka
    # offsety, by słupki stały obok siebie
    for r_ in range(R):
        # x-pozycja słupków dla klasy r:
        x_positions = index + (r_ - (R-1)/2)*bar_width
        ax.bar(x_positions,
               e_i_r[:, r_],
               bar_width,
               color=class_colors[r_],
               label=class_names[r_])

    ax.set_xlabel("Węzeł")
    ax.set_ylabel("e_{i,r}")
    ax.set_title("Prawdopodobieństwo odwiedzenia węzła")
    ax.set_xticks(index)  # centralnie
    ax.set_xticklabels(node_labels)
    ax.legend()
    ax.grid(True, axis='y')
    plt.show()

    # ---------------
    # Analogicznie możesz zmienić wykresy rho_{i,r}, K_{i,r}
    # by były "grouped" i mieć te same kolory + etykiety.
    # Przykład (nieco skrócony), np. dla rho_{i,r}:
    # ---------------

    rho_i_r = result['rho_i_r']
    fig, ax = plt.subplots(figsize=(8,5))
    for r_ in range(R):
        # Pozycje słupków danej klasy r_ (grouped bar)
        x_positions = index + (r_ - (R-1)/2)*bar_width
    
    # Ustawiamy etykietę w legendzie dla każdej klasy (bez żadnego warunku):
        ax.bar(
            x_positions,
            rho_i_r[:, r_],
            bar_width,
            color=class_colors[r_],
            label=class_names[r_]
        )
    ax.set_xlabel("Węzeł")
    ax.set_ylabel(r"$\rho_{i,r}$")
    ax.set_title("Obciążenia cząstkowe")
    ax.set_xticks(index)
    ax.set_xticklabels(node_labels)
    # aby w legendzie pojawiły się wszystkie klasy, usuń warunek "if r_==0".
    # lub stwórz unikatowe nazwy. Np. "label=class_names[r_]" w pętli, 
    # a na końcu: ax.legend() 
    ax.grid(True, axis='y')
    ax.legend()
    plt.legend()
    plt.show()

    # ---------------

    # (Przykład zmiany K_{i,r} na grouped bar - analogicznie)
    # (Przykład zmiany pi_i(k) - tam raczej mamy subplots, 
    #   więc kolory klas nie grają roli, bo to 1-wym. rozkład sumaryczny.)

    # Ostatni return:
    return result

