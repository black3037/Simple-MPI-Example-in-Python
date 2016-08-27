from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt

te=MPI.Wtime()

comm=MPI.COMM_WORLD
rank=comm.Get_rank()
size=comm.Get_size()
root = 0
last = size-1

L=10.0
Q=1.0
k=0.1
n=40
Delta=L/n

# determine work bounds
chunk=(n+1)/size
# give last bit of work to the last process
if rank==size-1:
    chunk+=n+1-chunk*size
# allocate local temperature arrays and loop bounds
if rank==0:
    T=np.zeros((n+1, chunk+1)) # one "ghost" column
    i_s, i_e = 0, chunk
elif rank==size-1:
    T=np.zeros((n+1, chunk+1)) # one "ghost" column
    i_s, i_e = 1, chunk+1
else:
    T=np.zeros((n+1, chunk+2)) # two "ghost" columns
    i_s, i_e = 1, chunk+1

# print "rank=%i i_s=%i i_e=%i chunk=%i" % (rank, i_s, i_e, chunk)

for iteration in range(10000):
    
    # save the old temperature
    T_old=1.0*T
    
    # the middle
    for j in range(1, n):
        for i in range(i_s+1, i_e-1):
            T[j,i]=0.25*Delta**2*Q/k+0.25*(T_old[j,i+1]+T_old[j,i-1]+T_old[j+1,i]+T_old[j-1,i])
    
    # boundaries
    i=i_s
    if rank==root:
        T[:,i],T[i,:],T[n,:]=0.0, 0.0, 0.0
    else :
        for j in range(0,n+1):
            if j==0 or j==n:
                T[j,:]=0.0
            else:
                T[j,i]=0.25*Delta**2*Q/k+0.25*(T_old[j,i+1]+T_old[j,i-1]+T_old[j+1,i]+T_old[j-1,i])
    i=i_e-1
    if rank==last:
        T[:,i],T[0,:],T[n,:]=0.0, 0.0, 0.0
    else :
        for j in range(0,n+1):
            if j==0 or j==n:
                T[j,:]=0.0
            else:
                T[j,i]=0.25*Delta**2*Q/k+0.25*(T_old[j,i+1]+T_old[j,i-1]+T_old[j+1,i]+T_old[j-1,i])
    
    # check for point-wise convergence
    ermax=np.zeros(n+1)
    for i in range(n+1):
        ermax[i]=max(abs(T[i,:]-T_old[i,:]))
    error=max(ermax)
    error=comm.reduce(error, error, op=MPI.MAX, root=root)
    error=comm.bcast(error,root=root)
    if rank==0 and iteration % 500 == 0:
        print "iteration %i, error %f" % (iteration, error)
    if error < 1e-6:
        break
    

    if rank > root:
        comm.send(T[:,i_s], dest=rank-1, tag=111)
    if rank < last:
        T[:,i_e]=comm.recv(source=rank+1, tag=111)
    # send right
    if rank < last:
        comm.send(T[:,i_e-1], dest=rank+1, tag=222)
    if rank > root:
        T[:,i_s-1]=comm.recv(source=rank-1, tag=222)


        
# get all temps on rank 0
T = comm.gather(T[:,i_s:i_e], root=root)

if rank ==0:

    te=MPI.Wtime()-te

    T_total = np.concatenate(T,axis=1)

    print T_total

    print "Elapsed time: %f seconds" % te
    print "Number of iterations: %i" % iteration
    print "Final error: %f" % error
    plt.contourf(T_total)
    plt.show()