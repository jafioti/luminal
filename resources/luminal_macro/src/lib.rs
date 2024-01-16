extern crate proc_macro;

use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, DeriveInput};

#[proc_macro_derive(LuminalEq)]
pub fn luminal_eq(input: TokenStream) -> TokenStream {
    // Parse the input tokens into a syntax tree
    let input = parse_macro_input!(input as DeriveInput);

    // Extract the ident and generics from the input
    let name = &input.ident;

    // Generics with expanded where clause
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    // Create the expanded trait implementation
    let expanded = quote! {
        impl #impl_generics core::cmp::PartialEq for #name #ty_generics #where_clause {
            fn eq(&self, _other: &Self) -> bool {
                false
            }
        }

        impl #impl_generics Eq for #name #ty_generics #where_clause {}
    };

    // Hand the output tokens back to the compiler
    TokenStream::from(expanded)
}

#[proc_macro_derive(LuminalPrint)]
pub fn luminal_print(input: TokenStream) -> TokenStream {
    // Parse the input tokens into a syntax tree
    let input = parse_macro_input!(input as DeriveInput);

    // Get the name of the struct
    let name = &input.ident;
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    // Create an identifier for the trait implementation
    let gen = quote! {
        impl #impl_generics std::fmt::Debug for #name #ty_generics #where_clause {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, stringify!(#name))
            }
        }
    };

    // Hand the generated implementation back to the compiler
    TokenStream::from(gen)
}
