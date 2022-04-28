


class Types_Analyse():

    def __init__():


        generate_data = Generate_Data(model,test_loader)
                activity_layer = generate_data.generate_preprocessed_data(multi = False)

                if type == 0:
                    print("CKA analysis")
                    if epoch == 0:
                        self.activity_layer_init = activity_layer
                        self.activity_layer_t_prime = activity_layer
                    else:
                        self.activity_layer_t = self.activity_layer_t_prime
                        self.activity_layer_t_prime = activity_layer

                        for layer in range(3):

                            cka = feature_space_linear_cka(self.activity_layer_t[layer], self.activity_layer_t_prime[layer], debiased=False)
                            cka_init =feature_space_linear_cka(self.activity_layer_init[layer], self.activity_layer_t_prime[layer], debiased=False)    
                            
                            for g in range(frequence_image_saving):
                                self.Liste_cka[layer].append(cka)
                                self.Liste_cka_init[layer].append(cka_init)

                if type == 1:
                    print("Isomap Plot")
                    representation_learning = Representation_Learning(frequence_image_saving, network_type)
                    representation_learning.Isomap_plot_save(activity_layer,epoch,network_type)