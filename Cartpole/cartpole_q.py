import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle
import winsound

def run(is_training=True, render=False, max_odul=1000, last_repeat=10):

    env = gym.make('CartPole-v1', render_mode='human' if render else None)

    # Bunlar sinirlandirmalar. Kisitlandirmak icin boyle yaptim, 10 parcaya boldum. https://gymnasium.farama.org/environments/classic_control/cart_pole/
    pos_space = np.linspace(-2.4, 2.4, 10)
    vel_space = np.linspace(-4, 4, 10)
    ang_space = np.linspace(-.2095, .2095, 10)
    ang_vel_space = np.linspace(-4, 4, 10)

    if(is_training):
        q = np.zeros((len(pos_space)+1, len(vel_space)+1, len(ang_space)+1, len(ang_vel_space)+1, env.action_space.n)) # 11x11x11x11x2 array olusturdum.
    else:
        f = open('cartpole.pkl', 'rb')
        q = pickle.load(f)
        f.close()

    learning_rate_a = 0.1 # ogrenme orani
    discount_factor_g = 0.99 # indirim faktoru

    epsilon = 1         # 1 = %100 rasgele hareket
    epsilon_decay_rate = 0.00001 # indirim fakturu
    rng = np.random.default_rng()

    rewards_per_episode = [] # Bolum basina oduller. 

    max_rewards = 0
    i = 0

    # for i in range(episodes):
    while(True):

        state = env.reset()[0]
        state_p = np.digitize(state[0], pos_space)
        state_v = np.digitize(state[1], vel_space)
        state_a = np.digitize(state[2], ang_space)
        state_av = np.digitize(state[3], ang_vel_space)

        terminated = False # Hedefe ulasirsa True olur.

        rewards=0

        while(not terminated and rewards < 1000000):

            if is_training and rng.random() < epsilon:
                # Rasgele yon sec.
                action = env.action_space.sample()
            else:
                # o inputlara sahip en iyi autput.
                action = np.argmax(q[state_p, state_v, state_a, state_av, :])

            new_state,reward,terminated,_,_ = env.step(action)
            new_state_p = np.digitize(new_state[0], pos_space)
            new_state_v = np.digitize(new_state[1], vel_space)
            new_state_a = np.digitize(new_state[2], ang_space)
            new_state_av= np.digitize(new_state[3], ang_vel_space)

            if is_training:
                # Q fonksyonu
                q[state_p, state_v, state_a, state_av, action] = q[state_p, state_v, state_a, state_av, action] + learning_rate_a * (
                    reward + discount_factor_g*np.max(q[new_state_p, new_state_v, new_state_a, new_state_av,:]) - q[state_p, state_v, state_a, state_av, action]
                )

            state = new_state
            state_p = new_state_p
            state_v = new_state_v
            state_a = new_state_a
            state_av= new_state_av

            rewards+=reward

            if not is_training and rewards%100==0:
                print(f'Episode: {i}  Rewards: {rewards}')

        rewards_per_episode.append(rewards) # Bolum basina oduller
        mean_rewards = np.mean(rewards_per_episode[len(rewards_per_episode)-100:]) # son 100 elemaninin ortalamasini al.

        if is_training and i%100==0:
            print(f'Tekrar: {i}  odul: {rewards}  Rasgelelik: {epsilon:03f}  Ortalama odul {mean_rewards}')

        if (mean_rewards>max_odul) and is_training: # Eger son 100 bolumun odulleri max_odul sayisindan buyukse islem biter.
            break

        if (mean_rewards>max_odul) and not is_training: # Eger son 100 bolumun odulleri max_odul sayisindan buyukse islem biter.
            if (last_repeat == i):
                break

        epsilon = max(epsilon - epsilon_decay_rate, 0)

        i+=1

    env.close()

    # Save Q table to file
    if is_training:
        f = open('cartpole.pkl','wb')
        pickle.dump(q, f)
        f.close()

    mean_rewards = []
    for t in range(i):
        mean_rewards.append(np.mean(rewards_per_episode[max(0, t-100):(t+1)]))
    plt.plot(mean_rewards)
    plt.savefig(f'cartpole.png')

    if is_training:

        frequency = 2500  
        duration = 1500  
        winsound.Beep(frequency, duration)

        new_last_repeat = last_repeat
        run(is_training=False, render=True, last_repeat=new_last_repeat)

if __name__ == '__main__':
    #run(is_training=True, render=False)

    traing = input("Bu bir eğtim mi? [(T)rue/(F)alse]: ")
    if traing == 'T' or traing == 't':
        new_traing = True
    else:
        if traing == 'F' or traing == 'f':
            new_traing = False
    
    if new_traing == True:
        odul = int(input("Maximum ödül kaç olsun? (Lütfen sayı girin) (Genelde 1000 yazılır) : "))
        old_render = False
        print("Eğitim bittikten sonra eğtim modeli kaydedilir ve eğitim sonu modeli 10 defa çalıştırılır.")
        how = 10
        
    if new_traing == False:
        how = int(input("Kaç defa göstereyim? (Lütfen sayı girin) (Genelde 10 yazılır) : "))
        old_render = True
        odul = 1000
    
    devam = input("Programı başlatmak için 'B' tuşuna ve ardından enter tuşuna basın. (Başka bir tuşa ve ardından enter tuşuna basarsanız program sonlandırılır) : ")

    if devam == 'B' or devam == 'b':
        run(is_training=new_traing, render=old_render, max_odul=odul, last_repeat=how)
    else:
        print("Program sonlandırıldı.")
        print("Güle güle...")