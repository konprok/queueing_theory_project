import numpy as np
from math import factorial, exp
import matplotlib.pyplot as plt

##########################
# 1) Budujemy 4D routing #
##########################

def build_routing_matrix_bcmp():
    """
    Tworzy 4-wymiarową macierz p_{(i, old_class)->(j, new_class)} = p_{(i, r)->(j, s)}
    w sieci 6-węzłowej:
      0=rejestracja, 1=admissions, 2=gynecology, 3=delivery, 4=icu, 5=postpartum.

    Klasy = 0,1,2 (zamiast 1,2,3). 
    Wartości prawdopodobieństw odwzorowują reguły z metody 'simulate(...)' w Twoim kodzie.
    """

    N = 6  # węzły
    R = 3  # klasy 0,1,2
    
    # Inicjalnie wszystko 0
    # routing_matrix[i, r, j, s] = p_{(i,r)->(j,s)}
    P = np.zeros((N, R, N, R))

    # --------------------------
    # (A) rejestracja (node 0)
    #
    # U Ciebie w symulacji: 
    #   pacjenci kl.1/2 => rejestracja => admissions
    #   pacjenci kl.3 (omijają rejestrację w symulacji).
    #
    # Przyjmijmy, że jeżeli klasa=2 => w rejestracji => 
    #   w 100% przechodzi do node1, klasa=2 (bez zmiany).
    #   klasa=0 => to samo, 
    #   klasa=2 => ...
    #   klasa=2 (czyli dawniej 3) w sumie w realu tu nie trafia. 
    #   W macierzy BCMP można wstawić p=1 do node1, kl.2, 
    #   lub p=0 jeżeli chcemy, by kl.2 w ogóle nie występowała w rejestracji.
    #
    # Dla uproszczenia: 
    #   p_{(0,0)->(1,0)}=1, p_{(0,1)->(1,1)}=1, p_{(0,2)->(1,2)}=1 (choć kl.2= '3' w Twoim kodzie
    #   faktycznie omija 0).
    #   W razie czego, i tak e_{0,2}=0 wyjdzie automatycznie, bo p0[0,2]=0 (patrz w BCMP).
    for r_ in [0,1,2]:
        P[0, r_, 1, r_] = 1.0

    # --------------------------
    # (B) izba przyjęć (node 1)
    #
    # z kodu:
    #   if class=0 => 10% out, 90% -> delivery(3) (bez zmiany klasy)
    #   if class=2 => 20% out, 40% -> delivery(3), 40% -> gyn(2)
    #   if class=1 => (czyli stara klasa=2 w poprzednim nazewnictwie?), w oryg. kl.3 => 100% do delivery
    #
    # Tu musimy pamiętać, że w Twoim kodzie "class"=3 => index=2. "class"=1 => index=0, "class"=2 => index=1.
    # Zatem:
    #   klasa=0 => out=0.1, do node3 klasa=0 =>0.9
    #   klasa=1 => out=0.2, do node3 klasa=1 =>0.4, do node2 klasa=1 =>0.4
    #   klasa=2 => 1.0 do node3 klasa=2
    #
    # Ponieważ w BCMP "wyjście" = brak kolejnego węzła => p=0 w macierzy. 
    # Niech out=10% to usuwamy te 10% (to się nie sumuje do 1 w P).
    # Zostawiamy tylko 0.9 do (3,0).
    # itd.
    # Reasumując:
    #  (1,0)->(3,0) =0.9
    #  (1,1)->(3,1)=0.4, (1,1)->(2,1)=0.4
    #  (1,2)->(3,2)=1.0
    P[1, 0, 3, 0] = 0.9  # klasa=0 => 90% do delivery
    P[1, 1, 3, 1] = 0.4  # klasa=1 => 40% do delivery
    P[1, 1, 2, 1] = 0.4  # klasa=1 => 40% do gyn
    P[1, 2, 3, 2] = 1.0  # klasa=2 => 100% do delivery

    # --------------------------
    # (C) ginekologia (2)
    #   if klasa=1 => 10% out, 90% -> delivery(3)
    #   (jeśli klasa=0..2 => tu w oryg. tylko klasa=1 wchodzi do gyn, 
    #    więc p=(2,0->? )=0, etc.)
    P[2, 1, 3, 1] = 0.9

    # --------------------------
    # (D) porodówka (3)
    # Tu następuje **class switching**. 
    #   stara klasa=2 => nowa klasa=1 z p=0.6, nowa=2 z p=0.3, nowa=2 =>3 z p=0.1
    #   ... itd. 
    #
    # W oryginalnym kodzie:
    #   if old_class=3 => new_class=2(0.4),1(0.3),3(0.3)
    #   if old_class=2 => new_class=1(0.6),3(0.1),2(0.3)
    #   if old_class=1 => new_class=3(0.03),2(0.07),1(0.9)
    #
    # Następnie, jeśli new_class=2 => idzie do OIOM(4),
    #  w innym wypadku (new_class=0 lub 1) => postpartum(5).
    #
    # Ale uwaga: my mamy "klasa"=0 => stary=1 => w oryg. 
    # Lepiej spisać to w sensie "old_class index -> new_class index".
    # Mamy mapę:
    #   old=2 => (new=1 p=0.4, new=0 p=0.3, new=2 p=0.3) w oryg. 
    # Ale to jest odwrotnie... Przetłumaczmy:
    #
    # Tabelka:
    #   oryginalnie: old=3 => new=2(0.4), new=1(0.3), new=3(0.3)
    #   w naszych indexach: old=2 => new=1(0.4), new=0(0.3), new=2(0.3)
    #
    # Podobnie:
    #   old=2 => new=1(0.6), new=3(0.1), new=2(0.3)
    #   => w indexach: old=1 => new=0(0.6), new=2(0.1), new=1(0.3)
    #
    #   old=1 => new=3(0.03), new=2(0.07), new=1(0.9)
    #   => w indexach: old=0 => new=2(0.03), new=1(0.07), new=0(0.9)
    #
    # Po ustaleniu nowej klasy:
    #   if new_class=2 => węzeł=4 (icu),
    #   else => węzeł=5 (postpartum).
    #
    # Zaimplementujmy to:
    
    # old=2 => [ new=1 p=0.4, new=0 p=0.3, new=2 p=0.3 ]
    #   => if new=2 => node=4, else => node=5
    P[3, 2, 4, 2] = 0.3  # old=2 -> new=2 => ICU
    P[3, 2, 5, 1] = 0.4  # old=2 -> new=1 => postpartum
    P[3, 2, 5, 0] = 0.3  # old=2 -> new=0 => postpartum
    
    # old=1 => [ new=0 p=0.6, new=2 p=0.1, new=1 p=0.3 ]
    P[3, 1, 4, 2] = 0.1
    P[3, 1, 5, 0] = 0.6
    P[3, 1, 5, 1] = 0.3
    
    # old=0 => [ new=2 p=0.03, new=1 p=0.07, new=0 p=0.9 ]
    P[3, 0, 4, 2] = 0.03
    P[3, 0, 5, 1] = 0.07
    P[3, 0, 5, 0] = 0.9

    # --------------------------
    # (E) OIOM (4)
    #   new_class = random(0 or 1) z p=0.5 => 
    #   if new_class=0 => postpartum (5),
    #   if new_class=1 => "other_department"? W oryg. "np.random.choice([1,2],p=[0.5,0.5])".
    #   Ale tam if new_class=1 => postpartum, else => out. 
    #   => zatem old_class cokolwiek => new_class=0 => postpartum, new_class=1 => out.
    #
    # W indexach: new_class=0 => postpartum, new_class=1 => "out" = brak węzła?
    # Ale Twój snippet mówi: "if new_class=1 => postpartum, else => other_department"...
    #   W python: np.random.choice([1,2], p=[0.5,0.5]) => 1 => postpartum, 2 => other
    #   w indexach: klasa=0 => postpartum, klasa=1 => out ?
    #
    # Przyjmijmy starannie:
    #   if wylosowano=1 => (co w indexach jest '0'?), i.e. musimy się zdecydować.
    #   Będzie prościej przypisać:
    #     wylosowano=1 -> new_class=0 => postpartum
    #     wylosowano=2 -> new_class=1 => out
    #
    # Zrobimy tak: Dla old_class= ANY(0..2):
    #   p_{(4,old)->(5,0)}= 0.5
    #   p=0 do (jakiegokolwiek node, cokolwiek) = 0.5 -> out?
    #   W BCMP "out" = brak wiersza w P => nic nie sumuje do 1. 
    # Więc p_{(4, old)-> ... }=0.5 do postpartum(5,0). 
    # Tyle że ... "postpartum(5, new_class=0)? klasa=0"? tak. 
    for old_c in [0,1,2]:
        P[4, old_c, 5, 0] = 0.5
        # te 0.5 do "out" nie pojawia się w macierzy, więc sum do 0.5
    
    # --------------------------
    # (F) sala poporodowa (5)
    #   if class=0 => out
    #   if class=1 => 70% => nowa kl=0 => out, else => kl=1 => out => generalnie out
    # W sumie zawsze out. Czyli 0% do kolejnych węzłów. 
    # => W BCMP to p=0, bo całość idzie "out".
    
    return P


#############################
# 2) BCMP liczenie (4D)     #
#############################

def _mmc_stationary_distribution(c, rho, max_k=20):
    """
    Zwraca listę wartości pi(k) dla k=0..max_k w systemie M/M/c
    z parametrem obciążenia 'rho' = λ/μ (zakładamy rho < c).
    """
    if rho >= c:
        return None

    # Normalizacja:
    #   Z = sum_{k=0..c-1} (rho^k / k!) + (rho^c / c!) * (c/(c-rho))
    #   p0=1/Z
    Z = 0.0
    from math import factorial
    for k in range(c):
        Z += (rho**k)/factorial(k)
    Z += (rho**c / factorial(c))*( c/(c-rho) )

    p0 = 1.0/Z
    dist=[]
    for k in range(max_k+1):
        if k<c:
            val = p0*(rho**k)/factorial(k)
        else:
            val = p0*(rho**c/factorial(c))*((1.0/c)**(k-c))
        dist.append(val)
    return dist

def compute_bcmp_open_network_4d(
    N, R, 
    arrival_rates,      # np.array(R,)
    service_rates,      # shape=(N,R)
    node_types,         # np.array(N,) in {1,2,3,4}
    servers_per_node,   # np.array(N,)
    routing_matrix_4d,  # shape=(N,R,N,R)
    max_k_for_print=10
):
    """
    BCMP z obsługą przełączania klas: p_{(i,r)->(j,s)}.

    Zwraca dict z kluczami:
     'e_i_r', 'rho_i_r', 'rho_i', 'K_i_r', 'K_i', 'pi_i_k' (rozkład stacjonarny)
    """
    # 1) p0[i,r] -> skąd klasa r wchodzi z zewnątrz?
    #  Dla uproszczenia: klasa 0,1 wchodzi do rejestracji (node0), klasa2 wchodzi do node1?
    #  (lub odwrotnie, zależy od definicji)
    p0 = np.zeros((N,R))
    # powiedzmy: klasa0 i1 => node0, klasa2 => node1
    # (odpowiada to temu, że klasa '3' w Twoim kodzie omija rejestrację)
    p0[0,0] = 1.0  # klasa0 => node0
    p0[0,1] = 1.0  # klasa1 => node0
    p0[1,2] = 1.0  # klasa2 => node1

    # 2) iteracyjne e_{i,r}
    e_i_r = np.zeros((N,R))
    e_i_r[:] = p0[:]
    for _iter in range(1000):
        new_e = np.copy(e_i_r)
        for i in range(N):
            for r_ in range(R):
                external_inflow = arrival_rates[r_]*p0[i,r_]
                internal_inflow = 0.0
                # p_{(j,s)->(i,r_)}
                for j in range(N):
                    for s in range(R):
                        internal_inflow += e_i_r[j,s]* routing_matrix_4d[j,s,i,r_]
                new_e[i,r_] = external_inflow + internal_inflow
        if np.allclose(new_e, e_i_r, atol=1e-12):
            break
        e_i_r = new_e

    # 3) ρ_{i,r} = (λ_r * e_{i,r}) / μ_{i,r}
    rho_i_r = np.zeros((N,R))
    for i in range(N):
        for r_ in range(R):
            mu_ = service_rates[i,r_]
            if mu_>0:
                rho_i_r[i,r_] = (arrival_rates[r_]* e_i_r[i,r_])/ mu_
    # sumaryczne obciążenie
    rho_i = np.sum(rho_i_r, axis=1)

    # 4) K_i_r
    K_i_r = np.zeros((N,R))
    for i in range(N):
        typ = node_types[i]
        if typ==1 or typ==4:  # M/M/1 or PS
            if rho_i[i]<1:
                for r_ in range(R):
                    if rho_i[i]>0:
                        K_i_r[i,r_] = (rho_i_r[i,r_]/rho_i[i])*( rho_i[i]/(1-rho_i[i]) )
            else:
                K_i_r[i,:]= np.inf
        elif typ==3:  # M/M/∞
            for r_ in range(R):
                K_i_r[i,r_] = rho_i_r[i,r_]
        elif typ==2:  # M/M/c
            c_ = servers_per_node[i]
            # lambd_i, mu_avg:
            lambd_i = 0.0
            mu_list=[]
            for r_ in range(R):
                lambd_i += arrival_rates[r_]* e_i_r[i,r_]
                if service_rates[i,r_]>0:
                    mu_list.append(service_rates[i,r_])
            if len(mu_list)==0:
                K_i_r[i,:]=0
                continue
            mu_avg= np.mean(mu_list)
            rho_prime = lambd_i/(c_*mu_avg)
            if rho_prime>=1.0:
                K_i_r[i,:]=np.inf
            else:
                # ErlangC
                def erlang_c(c,a):
                    num = (a**c)/ factorial(c)* (c/(c-a))
                    s_=0.0
                    for k_ in range(c):
                        s_+= (a**k_)/ factorial(k_)
                    s_+= num
                    p0= 1.0/s_
                    return num*p0
                a= lambd_i/mu_avg
                C= erlang_c(c_,a)
                Lq= C*(rho_prime/(1-rho_prime))
                Ls= c_*rho_prime
                L= Lq + Ls
                # Rozdział wg alpha_r= rho_i_r[i,r_]/ sum_rhoi
                sr= np.sum(rho_i_r[i,:])
                if sr>0:
                    for r_ in range(R):
                        alpha= rho_i_r[i,r_]/ sr
                        K_i_r[i,r_]= alpha*L
        else:
            # nic
            pass
    K_i= np.sum(K_i_r, axis=1)

    # 5) Rozkład stacjonarny pi_i(k)
    pi_i_k={}
    for i in range(N):
        typ= node_types[i]
        if typ==1 or typ==4:
            r_= rho_i[i]
            if r_<1.0:
                dist=[]
                for k_ in range(max_k_for_print+1):
                    dist.append( (1-r_)* (r_**k_) )
                pi_i_k[i]= np.array(dist)
            else:
                pi_i_k[i]=None
        elif typ==3:
            r_= rho_i[i]
            dist=[]
            for k_ in range(max_k_for_print+1):
                val= exp(-r_)* (r_**k_)/ factorial(k_)
                dist.append(val)
            pi_i_k[i]= np.array(dist)
        elif typ==2:
            # M/M/c
            c_= servers_per_node[i]
            # lambd_i, mu_avg
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
            big_rho= lambd_i/mu_avg
            if big_rho>= c_:
                pi_i_k[i]= None
            else:
                dist= _mmc_stationary_distribution(c_, big_rho, max_k_for_print)
                pi_i_k[i]= np.array(dist) if dist else None
        else:
            pi_i_k[i]= None

    return {
        'e_i_r': e_i_r,
        'rho_i_r': rho_i_r,
        'rho_i': rho_i,
        'K_i_r': K_i_r,
        'K_i': K_i,
        'pi_i_k': pi_i_k
    }

##################
# 3) DEMO        #
##################

if __name__=="__main__":
    # Mamy 6 węzłów, 3 klasy (0=kl1,1=kl2,2=kl3).
    N=6
    R=3
    
    # Zbudujmy routing z 4D:
    routing_4d= build_routing_matrix_bcmp()

    # Ustal intensywności zewn. λ_r. np. λ0=0.3, λ1=0.15, λ2=0.05
    arrival_rates= np.array([0.3, 0.15, 0.05])

    # node_types (rejestracja=1 serwer => M/M/1, izba przyjęć=2serw => M/M/c, gyn=2serw => M/M/c..., 
    # ... w Twoim config = niemal wszystko M/M/c oprócz rejestracji?)
    # Dla przykładu:
    node_types= np.array([1,2,2,2,2,2])  # rejestracja -> type=1, reszta -> type=2
    
    # servers:
    servers_per_node= np.array([1,2,10,2,3,10], dtype=int)
    
    # service_rates[i,r]: 
    # np. rejestracja(0) mu=4.0, izba(1) mu=3.0, gyn(2) mu=0.02/h ???, etc.
    service_rates= np.zeros((N,R))
    service_rates[0,:]= 4.0     # rejestracja
    service_rates[1,:]= 3.0     # admissions
    service_rates[2,:]= 0.02    # gyn
    service_rates[3,:]= 0.33    # delivery
    service_rates[4,:]= 0.04    # icu
    service_rates[5,:]= 0.028   # postpartum

    result= compute_bcmp_open_network_4d(
        N= N,
        R= R,
        arrival_rates= arrival_rates,
        service_rates= service_rates,
        node_types= node_types,
        servers_per_node= servers_per_node,
        routing_matrix_4d= routing_4d,
        max_k_for_print=20
    )

    print("\n=== WYNIKI BCMP (z class switching) ===\n")
    print("e_{i,r}:")
    print(result['e_i_r'])

    print("\nrho_{i,r} (cząstkowe obciążenia):")
    print(result['rho_i_r'])

    print("\nrho_i (suma obciążeń w węźle i):")
    print(result['rho_i'])

    print("\nK_i_r (średnia liczba zadań klasy r w węźle i):")
    print(result['K_i_r'])

    print("\nK_i (suma wszystkich klas w węźle i):")
    print(result['K_i'])

    print("\npi_i(k) – rozkłady stacjonarne (dla k=0..20):")
    for i in range(N):
        print(f"Węzeł {i}, typ={node_types[i]}:", result['pi_i_k'][i])

    # Na koniec, przykładowy "wykres" obciążeń:
    plt.figure()
    plt.bar(range(N), result['rho_i'], color='orange', alpha=0.7)
    plt.xlabel("Węzeł i")
    plt.ylabel("Obciążenie ρ_i")
    plt.title("Sumaryczne obciążenia węzłów (BCMP)")
    plt.grid(True)
    plt.show()
