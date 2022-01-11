# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 16:22:00 2019

@author: Artur Yakimovich, PhD (c) 2019. UCL
"""

def dice_coeff_standard(yt, yp):

    yt = yt > 0.5
    yp = yp > 0.5

    return 2*np.sum(np.logical_and(yt, yp)) / (np.sum(yt) + np.sum(yp))


def predict():

    model = get_unet()

    print('- ' * 30)
    print('Loading and preprocessing test data...')
    print('- ' * 30)

    test_data = np.load('data/DRIVE/imgs_test2.npz')
    imgs_test, imgs_mask_test = test_data['imgs'], test_data['imgs_mask']

    imgs_test = imgs_test.astype('float32')
    imgs_test = imgs_test / imgs_test.max()
    # imgs_test -= mean
    # imgs_test /= std

    print('- ' * 30)
    print('Loading saved weights...')
    print('- ' * 30)
    #model.load_weights('models/modelaug1-3900.h5')
    model.load_weights('models/modelaug2-2000.h5')

    print('- ' * 30)
    print('Predicting masks on test data...')
    print('- ' * 30)

    imgs_mask_pred = model.predict(imgs_test[:5], verbose=1)

#     np.save('imgs_mask_test.npy', imgs_mask_test)

    dice_all = []

    for impred, im, imtest in zip(imgs_mask_pred[:5], imgs_test[:5], imgs_mask_test[:5]):

        dice_all.append(dice_coeff_standard(imtest, impred))

        plt.figure(figsize=(9,9))
        plt.title("Ground truth and prediction for test set. Dice {}".format(dice_all[-1]))

        plt.subplot(1, 3, 1)
        plt.imshow(np.squeeze(im))
        plt.subplot(1, 3, 2)
        plt.imshow(np.squeeze(imtest))
        plt.subplot(1, 3, 3)
        plt.imshow(np.squeeze(impred))

    plt.show()

    print("Mean dice: ", np.mean(dice_all))