import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import { HciAtTuftsComponent } from './hci-at-tufts/hci-at-tufts.component';
import { PeopleComponent } from './people/people.component';
import { ProjectsComponent } from './projects/projects.component';
import { PublicationsComponent } from './publications/publications.component';
import { CodeAndDatasetsComponent } from './code-and-datasets/code-and-datasets.component';
import { AdmissionsComponent } from './admissions/admissions.component';

const routes: Routes = [
  { path: 'hci_at_tufts', component: HciAtTuftsComponent },
  { path: 'people', component: PeopleComponent },
  { path: 'projects', component: ProjectsComponent },
  { path: 'publications', component: PublicationsComponent },
  { path: 'code_and_datasets', component: CodeAndDatasetsComponent },
  { path: 'admissions', component: AdmissionsComponent },
];
@NgModule({
  imports: [RouterModule.forRoot(routes, {
    useHash: false
  })],
  exports: [RouterModule]
})
export class AppRoutingModule { }
