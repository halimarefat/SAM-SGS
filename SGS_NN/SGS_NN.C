/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | Copyright (C) 2011-2015 OpenFOAM Foundation
     \\/     M anipulation  |
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

\*---------------------------------------------------------------------------*/

#include <torch/torch.h>
#include <torch/script.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include "SGS_NN.H"
#include "fvOptions.H"
#include "wallDist.H"
#include "model.H"
#include "CustomDataset.H"
#include <vector>
#include <memory>
#include "string"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
namespace LESModels
{

torch::Tensor forward_on_multiple_gpus(torch::jit::script::Module& model, torch::Tensor input, const std::vector<torch::Device>& devices) 
{
    std::vector<torch::Tensor> outputs;
    auto chunked_inputs = input.chunk(devices.size(), 0); 

    for (size_t i = 0; i < devices.size(); ++i) 
    {
        //std::cout << "Before forward" << std::endl;
        auto device = devices[i];
        model.to(torch::kDouble);
        model.to(device);
        
        //Info << "+--- torch model is loaded." << nl;
        c10::IValue ifeat = c10::IValue(chunked_inputs[i].to(device)); 
        auto output = model.forward({ifeat}).toTensor();
        //auto output = model.forward({chunked_inputs[i].to(device)}).toTensor();  
        outputs.push_back(output.to(torch::kCPU));  
        //std::cout << "After forward" << std::endl;

        c10::cuda::CUDACachingAllocator::resetAccumulatedStats(i);
        c10::cuda::CUDACachingAllocator::resetPeakStats(i);
        c10::cuda::CUDACachingAllocator::emptyCache();
    }

    return torch::cat(outputs, 0);  
}

// * * * * * * * * * * * * * Private Member Functions  * * * * * * * * * * * //
template<class BasicTurbulenceModel>
void SGS_NN<BasicTurbulenceModel>::correctNut()
{
    if (!this->turbulence_)
    {
        return;
    }

    LESeddyViscosity<BasicTurbulenceModel>::correct();

    Foam::IOdictionary nnProperties
    (
        Foam::IOobject
        (
            "nnProperties",
            Foam::fileName(this->runTime_.constant()),
            this->mesh_,
            Foam::IOobject::MUST_READ,
            Foam::IOobject::NO_WRITE
        )
    );

    int64_t MNum = Foam::readLabel(nnProperties.lookup("MNum"));

    const int ARRAY_SIZE = 19;
    std::vector<float> means(ARRAY_SIZE);
    std::vector<float> stds(ARRAY_SIZE);

    std::vector<std::string> keys = {"G1","G2","G3","G4","G5","G6",
                                     "S1","S2","S3","S4","S5","S6",
                                     "UUp1","UUp2","UUp3","UUp4","UUp5","UUp6", 
                                     "Nut"};
    // Read values
    for (int i = 0; i < ARRAY_SIZE; ++i)
    {
        means[i] = Foam::readScalar(nnProperties.subDict("means").lookup(Foam::word(keys[i])));
        stds[i] = Foam::readScalar(nnProperties.subDict("stds").lookup(Foam::word(keys[i])));
    }

    Info << "----**** means 0: " << means[0] << nl;
    Info << "----**** means 1: " << means[1] << nl;

    Info << "----**** stds 0: " << stds[0] << nl;
    Info << "----**** stds 1: " << stds[1] << nl;

    volScalarField u_ = this->U_.component(vector::X);
    volScalarField v_ = this->U_.component(vector::Y);
    volScalarField w_ = this->U_.component(vector::Z);

    volTensorField G = fvc::grad(this->U_);
    volScalarField G11 = G.component(tensor::XX);
    volScalarField G12 = G.component(tensor::XY);
    volScalarField G13 = G.component(tensor::XZ);
    volScalarField G22 = G.component(tensor::YY);
    volScalarField G23 = G.component(tensor::YZ);
    volScalarField G33 = G.component(tensor::ZZ);

    volSymmTensorField S(dev(symm(G)));
    volScalarField S11 = S.component(tensor::XX);
    volScalarField S12 = S.component(tensor::XY);
    volScalarField S13 = S.component(tensor::XZ);
    volScalarField S22 = S.component(tensor::YY);
    volScalarField S23 = S.component(tensor::YZ);
    volScalarField S33 = S.component(tensor::ZZ);

    volSymmTensorField UUp( IOobject("UPrime2Mean", this->runTime_.timeName(), this->mesh_, IOobject::READ_IF_PRESENT), symm(G));
    //volSymmTensorField UUp = createField<volSymmTensorField>(this->runTime_,this->mesh_,"UPrime2Mean");
    volScalarField UUp11 = UUp.component(tensor::XX);
    volScalarField UUp12 = UUp.component(tensor::XY);
    volScalarField UUp13 = UUp.component(tensor::XZ);
    volScalarField UUp22 = UUp.component(tensor::YY);
    volScalarField UUp23 = UUp.component(tensor::YZ);
    volScalarField UUp33 = UUp.component(tensor::ZZ);

    int64_t in_s = -3999;
    int64_t ot_s = -3999;
    
    std::vector<std::vector<double>> in_data;
    forAll(u_, i)
    {
        std::vector<double> tmp;
        if(MNum==2)
        {
            in_s = 12;
            ot_s = 1;
            tmp.push_back((G11[i]-means[0])/stds[0]);
            tmp.push_back((G12[i]-means[1])/stds[1]);
            tmp.push_back((G13[i]-means[2])/stds[2]);
            tmp.push_back((G22[i]-means[3])/stds[3]);
            tmp.push_back((G23[i]-means[4])/stds[4]);
            tmp.push_back((G33[i]-means[5])/stds[5]);
            tmp.push_back((S11[i]-means[6])/stds[6]);
            tmp.push_back((S12[i]-means[7])/stds[7]);
            tmp.push_back((S13[i]-means[8])/stds[8]);
            tmp.push_back((S22[i]-means[9])/stds[9]);
            tmp.push_back((S23[i]-means[10])/stds[10]);
            tmp.push_back((S33[i]-means[11])/stds[11]);
        }
        else if(MNum==5)
        {
            in_s = 18;
            ot_s = 1;
            tmp.push_back((G11[i]-means[0])/stds[0]);
            tmp.push_back((G12[i]-means[1])/stds[1]);
            tmp.push_back((G13[i]-means[2])/stds[2]);
            tmp.push_back((G22[i]-means[3])/stds[3]);
            tmp.push_back((G23[i]-means[4])/stds[4]);
            tmp.push_back((G33[i]-means[5])/stds[5]);
            tmp.push_back((S11[i]-means[6])/stds[6]);
            tmp.push_back((S12[i]-means[7])/stds[7]);
            tmp.push_back((S13[i]-means[8])/stds[8]);
            tmp.push_back((S22[i]-means[9])/stds[9]);
            tmp.push_back((S23[i]-means[10])/stds[10]);
            tmp.push_back((S33[i]-means[11])/stds[11]);
            tmp.push_back((UUp11[i]-means[12])/stds[12]);
            tmp.push_back((UUp12[i]-means[13])/stds[13]);
            tmp.push_back((UUp13[i]-means[14])/stds[14]);
            tmp.push_back((UUp22[i]-means[15])/stds[15]);
            tmp.push_back((UUp23[i]-means[16])/stds[16]);
            tmp.push_back((UUp33[i]-means[17])/stds[17]);
        }

        in_data.push_back(tmp);
    }
    
    const int64_t batchSize = Foam::readLabel(nnProperties.lookup("Batch_Size"));
    const int64_t batchNum = in_data.size() / batchSize;
    Info << "+--- batch size: " << batchSize << nl;

    auto ds  = CustomDataset(in_data, in_s).map(torch::data::transforms::Stack<>());
    auto dsloader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>
                          ( std::move(ds), batchSize);
    //Info << "+--- data loader is ready." << nl;

    std::string device = Foam::word(nnProperties.lookup("device"));
    Foam::fileName torchModelPath(nnProperties.lookup("torchModelPath"));

    torch::DeviceType deviceType;
    if (device == "cuda") 
    {
        deviceType = torch::kCUDA;
        Info << "+--- device is: " << device << nl;
    } 
    else if (device == "cpu")
    {
        deviceType = torch::kCPU;
        Info << "+--- device is: " << device << nl; 
    }
    
    std::vector<torch::Device> devices;
    for (int i = 0; i < torch::cuda::device_count(); ++i) 
    {
        devices.push_back(torch::Device(torch::kCUDA, i));
    }

    torch::jit::script::Module torchModel = torch::jit::load(torchModelPath.c_str());
    //torchModel.to(deviceType);
    //torchModel.to(torch::kDouble);
    //Info << "+--- torch model is loaded." << nl;

    torch::Tensor pred;
    int64_t i = 0;
    torch::NoGradGuard no_grad;
    for(torch::data::Example<>& batch : *dsloader)
    {
        Info << "batch " << i++ << " of " << batchNum << nl;
        auto feat = batch.data.to(torch::kDouble).to(deviceType);
        //c10::IValue ifeat = c10::IValue(feat); 
        //std::cout << "+--- feat: " << feat[0] << std::endl;
        //auto output = torchModel.forward({ifeat});
        //auto batch_pred = output.toTensor();
        torch::Tensor batch_pred = forward_on_multiple_gpus(torchModel, feat, devices);

        //Info << "batch size: " << batch_pred.numel() << nl;
        if (pred.numel() == 0) 
        {
            pred = batch_pred;  
        } 
        else 
        {
            pred = torch::cat({pred, batch_pred}, 0);  
        }
        
    }
    
    Info << "prediction is done!" << nl;
    forAll(this->nut_, i)
    {
        float value = pred[i].item<float>() * stds[18] + means[18];
        // Check for NaN
        if (std::isnan(value))
        {
            Info << "Warning: NaN detected in nut_ at index " << i << nl;
            value = 0.0000001; // or assign a safe default value
        }

        if (std::isinf(value))
        {
            Info << "Warning: Infinite value detected in nut_ at index " << i << nl;
            value = 1.0; // or another appropriate default value
        }
        // Check for zero or other unwanted values
        if (value == 0.0)
        {
            Info << "Warning: Zero detected in nut_ at index " << i << nl;
            value = 0.0000001;
            // Handle it as per your requirement, e.g., value = epsilon;
        }

        // Assign the value
        this->nut_[i] = value;
    }

    this->nut_.correctBoundaryConditions();
    fv::options::New(this->mesh_).correct(this->nut_);
    BasicTurbulenceModel::correctNut();

}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

template<class BasicTurbulenceModel>
SGS_NN<BasicTurbulenceModel>::SGS_NN
(
    const alphaField& alpha,
    const rhoField& rho,
    const volVectorField& U,
    const surfaceScalarField& alphaRhoPhi,
    const surfaceScalarField& phi,
    const transportModel& transport,
    const word& propertiesName,
    const word& type
)
:
    LESeddyViscosity<BasicTurbulenceModel>
    (
        type,
        alpha,
        rho,
        U,
        alphaRhoPhi,
        phi,
        transport,
        propertiesName
    ),
    simpleFilter_(U.mesh()),
    filterPtr_(LESfilter::New(U.mesh(), this->coeffDict())),
    filter_(filterPtr_())
{
    if (type == typeName)
    {
        this->printCoeffs(type);
    }
}

// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

template<class BasicTurbulenceModel>
bool SGS_NN<BasicTurbulenceModel>::read()
{
    if (LESeddyViscosity<BasicTurbulenceModel>::read())
    {
        filter_.read(this->coeffDict());        

        return true;
    }
    else
    {
        return false;
    }
}

template<class BasicTurbulenceModel>
tmp<volScalarField> SGS_NN<BasicTurbulenceModel>::k
(
    const volTensorField& gradU
) const
{
    return tmp<volScalarField>
    (
        new volScalarField
        (
            IOobject
            (
                IOobject::groupName("k", this->U_.group()),
                this->runTime_.timeName(),
                this->mesh_
            ),
            sqr(this->nut_ / (this->delta() * 0.1))
        )
    );
}


template<class BasicTurbulenceModel>
void SGS_NN<BasicTurbulenceModel>::correct()
{
    correctNut();
}


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace LESModels
} // End namespace Foam

// ************************************************************************* //