rng('shuffle');

load tmp_generator4.mat
load tmp_discriminator4.mat

rand_z = rand([200,10],'single');
        
generator = myGenerator(generator,rand_z,'forward');
gen_output = generator.layers{6}.input{1};

discriminator = myDiscriminator(discriminator,gen_output,'forward','true');
disc_output_G = discriminator.layers{6}.input{1};

d_gen_Loss = (-1)*disc_output_G.^(-1);
discriminator = myDiscriminator(discriminator,d_gen_Loss,'backward','true');