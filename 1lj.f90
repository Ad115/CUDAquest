!Dinamica molecular en 2D Lenard-Jones
!Modulo para definir las variables, el LJ solo nos importa 
!la posicion de las partiulas, su velocidad y la fuerza de interaccion
!ademas debemos tener en cuenta sus variables termodinamicas (N,V,T)

	module variables
	implicit none 
	integer nat, npasos 
	double precision, dimension(:), allocatable :: rx , ry
	double precision, dimension(:), allocatable :: vx , vy
	double precision, dimension(:), allocatable :: fx , fy
	double precision :: lx, ly, rho 
	double precision :: sigma, eps, dt, rcut, upot, ukin, Tins, Temp 
	end 

!Este sera el programa principal 
	program Dinamica
	use variables

	implicit none  

	call inicio
	call posiciones 
!	call foto
	call velocidades
	end

!Vamos a llenar nuestras variables	
	subroutine inicio 
	use variables
	implicit none
	open(1,file='run.txt', status='old', action='read')
	 read(1,*)nat
	 read(1,*)rho
	 read(1,*)lx, ly
	 read(1,*)sigma, eps 
	 read(1,*)npasos
	 read(1,*)Temp
	close(1)
!Definimos el tamaño de nuestros arreglos
	allocate(rx(nat),ry(nat))
	allocate(rx(nat),ry(nat))
	allocate(rx(nat),ry(nat))
	end

!Para dar las posiciones tenemos que tomar en cuenta la densidad
!las dimensiones de la caja y la temperatura 
	subroutine posiciones 
	use variables
	implicit none
	
	integer :: i, j
	double precision :: dx, dy, rij, rndx, rndy

	do i=1, nat
	100 call random_number(rndx)
	  call random_number(rndy)
!Le damos una posicion aleatoria a las particulas dentro de la caja
	  rx(i)= rndx*lx
	  ry(i)= rndy*ly
!Verificamos que no haya traslape
	  do j=1, i-1 
	    dx= rx(i)-rx(j)
	    dy= ry(i)-ry(j)
!Condiciones periodicas con las cajas imagen 
	    if (dx>0.50*lx)then
		dx= dx-lx
	    elseif (dx<-0.50*lx)then
		dx= dx+lx
	    endif	

	    if (dy>0.50*ly)then
		dy= dy-ly
	    elseif (dy<-0.50*ly)then
		dy= dy+ly
	    endif
	    rij=sqrt(dx**2 + dy**2)
	    if (rij < sigma) then 
		go to 100
	    endif		
	  enddo
	enddo
	end

!Para ver una foto de lo que esta pasando
	subroutine foto
	use variables
	implicit none 

	integer :: i
	
	open(2,file='xyz.xyz')
	 write(2,*)nat
	 write(2,*)
	 do i=1,nat 
	  write(2,'(a,3f12.6)')'N', rx(i), ry(i), 0.0
	 enddo
	close(2)
	end

!Vamos a calcular las velocidades de las particulas
	subroutine velocidades
	use variables
	implicit none 
	
	integer :: i
	double precision :: rndx, rndy

	 do i=1,nat 
	 call random_number(rndy)
	 call random_number(rndx)

	 vx(i)=(2.0*rndx-1.0)
	 vy(i)=(2.0*rndy-1.0)
	 enddo
	end 

!En esta parte calcularemos la fuerza entre particulas 
!En este paso es donde introdujimos el potencial de Lenard-Jones para 
!calcular la fuerza 
	subroutine fuerza
	use variables
	implicit none 

	integer :: i,j 
	double precision :: dx, dy, rij, duij

	fx=0.0
	fy=0.0
	upot=0.0
	ukin=0.0
	
	do i=1,nat-1
 	 do j=i+1,nat		
	     dx =rx(i)-rx(j)
	     dy =ry(i)-ry(j)
!Condiciones periodicas con las cajas imagen 
	    if (dx>0.50*lx)then
		dx= dx-lx
	    elseif (dx<-0.50*lx)then
		dx= dx+lx
	    endif	

	    if (dy>0.50*ly)then
		dy= dy-ly
	    elseif (dy<-0.50*ly)then
		dy= dy+ly
	    endif
	     rij= sqrt(dx**2+dy**2)
!Radio de corte
	     if(rij<rcut)then
!energia potencial
	     upot=upot + 4.0*eps*(sigma/rij)**6 *((sigma/rij)**6 -1.0)
	     duij=24.0*eps*(sigma/rij)**6 *((2.0*sigma/rij)**6 -1.0)/rij
	     fx(i)=fx(i) + duij*dx/rij
	     fy(i)=fy(i) + duij*dy/rij
	     fx(j)=fx(j) - duij*dx/rij
	     fx(j)=fx(j) - duij*dy/rij
	     endif	
	 enddo
	enddo
	end

!En esta parte es donde le damos la dinamica al sistema 
	subroutine mdloop
	use variables 
	implicit none 

	integer :: paso, i
	
	ukin=0.0

	do paso=1, npasos
	  do i=1, nat
	    vx(i)=vX(i) + fx(i)*dt*0.5
	    vy(i)=vy(i) + fy(i)*dt*0.5
	    rx(i)=rx(i)+vx(i)*dt	
	    ry(i)=ry(i)+vy(i)*dt
!Condiciones periodicas para mantener el mismo 
!numero de particulas en la caja
	    if (rx(i)>lx) rx(i) = rx(i) - lx
	    if (ry(i)>ly) ry(i) = ry(i) - ly
	    if (rx(i)<lx) rx(i) = rx(i) + lx
	    if (ry(i)<ly) ry(i) = ry(i) + ly
	  enddo
	call fuerza
	 do i=1,nat
	  vx(i) = vx(i) + fx(i)*dt*0.5
	  vy(i) = vy(i) + fy(i)*dt*0.5
	  ukin= ukin + vx(i)**2 + vy(i)**2
	 enddo
	Tins=ukin/nat
	ukin=0.5*ukin	
	do
	vx(i)=vx(i)*sqrt(Temp/Tins)
	vy(i)=vy(i)*sqrt(Temp/Tins)
	enddo
	
	if (mod(paso,100)==0)then
	  write(2,*)nat
	  write(2,*)
	  do i=1,nat
		write(2,'(a,3f12.6)')'N', rx(i), ry(i), 0.0
	  enddo
	endif
	enddo 

	end


